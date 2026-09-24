'''
Modulo (g): analisis de generalizacion a tipos de defecto no vistos (R6).

QUE MIDE Y POR QUE ES POSIBLE MEDIRLO
-------------------------------------
La particion de R4 reserva en cada categoria de MVTec AD un tipo de defecto
COMPLETO, sorteado con semilla fija, que no aporta ninguna imagen al
subconjunto etiquetado de entrenamiento y permanece integro en el conjunto de
prueba. Esa reserva es lo que hace verificable el resultado R6: sin ella, todos
los tipos presentes en la prueba habrian estado representados de algun modo en
el entrenamiento y no existiria ningun tipo "no visto" sobre el que medir.

El analisis parte en dos el conjunto de prueba, MANTENIENDO LAS MISMAS
IMAGENES CONFORMES en ambas mitades:

    vistos     = conformes de prueba + anomalas de tipos SI representados
    reservado  = conformes de prueba + anomalas del tipo NO representado

Compartir los negativos no es un detalle de implementacion: si cada mitad
usara conformes distintas, la diferencia entre ambas AUROC mezclaria el efecto
del tipo de defecto con el de haber cambiado la clase conforme, y dejaria de
ser atribuible a la generalizacion.

CRITERIO DEL R6 (IOV declarado)
-------------------------------
Detectar correctamente al menos el 90 % de las anomalias pertenecientes al tipo
no representado. Es una tasa de RECALL, no un AUROC, de modo que exige un punto
de operacion: se emplea el mismo umbral con el que se reporta la categoria en
R5 -el calibrado con las dos clases de entrenamiento (`umbral_balanceado`)-,
no uno elegido para este analisis. Elegir aqui un umbral distinto convertiria
el indicador en una medida de la calibracion y no de la generalizacion.

LA OBJECION QUE HAY QUE ADELANTARSE A RESPONDER
-----------------------------------------------
Que el tipo reservado rinda peor que los vistos admite dos lecturas: que el
modelo generaliza mal, o que a ese tipo concreto le toco ser el mas dificil de
su categoria con independencia de lo que haya visto el entrenamiento. Las dos
predicen lo mismo sobre la mitad reservada, asi que el contraste vistos/
reservado por si solo NO distingue entre ellas.

Por eso se reporta ademas el AUROC de CADA tipo de defecto por separado, junto
con el numero de imagenes que aporto al entrenamiento. Eso permite situar al
tipo reservado dentro de la distribucion de dificultad de su propia categoria:
si su rendimiento cae dentro del rango de los tipos vistos, la hipotesis del
"tipo dificil" queda sin apoyo; si queda sistematicamente el ultimo en muchas
categorias a la vez, el azar deja de ser una explicacion economica. El campo
`rango_reservado` del informe recoge exactamente esa posicion.

ALCANCE
-------
Once categorias de MVTec AD. `toothbrush` queda fuera porque tiene un unico
tipo de defecto y reservarlo dejaria vacio su subconjunto etiquetado. VisA
queda fuera por completo: el corpus no clasifica sus anomalias por tipo ni en
la estructura de directorios ni en sus ficheros de particion oficiales, de modo
que la frontera "visto / no visto" no puede definirse. Ambas limitaciones son
del corpus y de la disponibilidad de anotacion, no del procedimiento.

USO
---
El desglose se calcula DENTRO del run (`modelo_semisupervisado.principal` lo invoca al terminar
cada categoria), porque necesita los mapas de pixeles, que no se persisten. El
informe agregado se genera despues, sin GPU:

    .\\venv\\Scripts\\python.exe -m modelo_semisupervisado.generalizacion
'''
import csv
import os

import numpy as np
from sklearn.metrics import roc_auc_score

# El directorio data/ vive DENTRO del paquete desde la reorganizacion del
# 2026-09-13; subir un nivel mas apuntaba a tesis/data/, que ya no existe.
RAIZ = os.path.dirname(os.path.abspath(__file__))
MANIFIESTO = os.path.join(RAIZ, "data", "manifiesto.csv")
RESUMEN = os.path.join(RAIZ, "data", "resumen.csv")

# Umbral del IOV de R6: fraccion minima de anomalias del tipo no representado
# que el sistema debe detectar en el punto de operacion declarado.
RECALL_OBJETIVO = 0.90

CAMPOS_CAT = ("categoria", "reservado", "n_test_good", "n_vistos", "n_reservado",
              "auroc_img_vistos", "auroc_img_reservado", "delta_img",
              "auroc_pix_vistos", "auroc_pix_reservado", "delta_pix",
              "recall_vistos", "recall_reservado", "fpr_good", "umbral",
              "rango_reservado", "n_tipos")
CAMPOS_TIPO = ("categoria", "tipo", "reservado", "n_train", "n_test",
               "auroc_img", "auroc_pix", "recall")


# %% lectura de la particion

_cache_manifiesto = None


def _rel(ruta_fichero):
    '''Ruta relativa a la categoria, con el separador normalizado.

    El manifiesto guarda `test/anomaly/<fichero>`; en Windows las rutas del
    dataset llegan con barra invertida. Se normaliza y se toman los tres
    ultimos componentes, que son exactamente split/subcarpeta/fichero.
    '''
    partes = ruta_fichero.replace("\\", "/").split("/")
    return "/".join(partes[-3:])


def cargar_manifiesto(ruta=MANIFIESTO):
    '''{(dataset, categoria, archivo_rel): tipo_defecto} + recuentos por tipo.

    El manifiesto es la fuente de verdad del tipo de defecto. Deducirlo del
    nombre del fichero seria ambiguo -`broken_large_000.png` admite el corte
    `broken` + `large_000` tan bien como `broken_large` + `000`- y ademas
    duplicaria en el codigo de evaluacion una decision que pertenece a R4.
    '''
    global _cache_manifiesto
    if _cache_manifiesto is not None:
        return _cache_manifiesto
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            "No se encuentra el manifiesto de la particion en %s.\n"
            "Generalo con:  .\\venv\\Scripts\\python.exe data\\construir_particion.py" % ruta)
    tipos, conteo = {}, {}
    with open(ruta, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["etiqueta"] != "anomaly":
                continue
            clave = (r["dataset"], r["categoria"], r["archivo"])
            tipos[clave] = r["tipo_defecto"]
            k = (r["dataset"], r["categoria"], r["tipo_defecto"], r["split"])
            conteo[k] = conteo.get(k, 0) + 1
    _cache_manifiesto = (tipos, conteo)
    return _cache_manifiesto


def reservado_de(conjunto_datos, categoria, ruta=RESUMEN):
    '''Tipo de defecto reservado de una categoria, o None si no hay ninguno.'''
    if not os.path.exists(ruta):
        return None
    with open(ruta, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset"] == conjunto_datos and r["categoria"] == categoria:
                v = r["reservado"].strip()
                return v if v and v != "-" else None
    return None


def tipos_de_muestras(muestras, conjunto_datos, categoria):
    '''Tipo de defecto de cada muestra del conjunto de prueba, en su orden.

    `muestras` es la lista (ruta, etiqueta, mascara) del Dataset. Las conformes
    devuelven cadena vacia. Si una anomala no aparece en el manifiesto se
    levanta el error en vez de silenciarlo: significa que las imagenes en disco
    y el manifiesto describen particiones distintas, y cualquier metrica
    desglosada calculada sobre esa mezcla careceria de sentido.
    '''
    tabla, _ = cargar_manifiesto()
    salida = []
    for ruta, etiqueta, _m in muestras:
        if etiqueta == 0:
            salida.append("")
            continue
        clave = (conjunto_datos, categoria, _rel(ruta))
        if clave not in tabla:
            raise KeyError(
                "%s no figura como anomala en el manifiesto. El directorio de "
                "la particion y data/manifiesto.csv no se corresponden: "
                "regenera la particion antes de evaluar." % _rel(ruta))
        salida.append(tabla[clave])
    return salida


# %% metricas desglosadas

def _auroc_imagen(puntajes, etiquetas, indice):
    y = np.asarray(etiquetas, dtype=int)[indice]
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(puntajes, dtype=float)[indice]))


def _auroc_pixel(mapas_pixeles, mascaras, indice):
    '''AUROC de pixel agrupando (pooled) los pixeles del subconjunto.

    Se agrupan los pixeles y se calcula UNA sola curva, igual que en la metrica
    global del modulo (e); promediar AUROC por imagen daria otro numero y no
    seria comparable con lo que reporta la literatura.
    '''
    m = np.concatenate([np.asarray(mascaras[i], dtype=np.uint8).ravel() for i in indice])
    if m.min() == m.max():
        return float("nan")
    p = np.concatenate([np.asarray(mapas_pixeles[i], dtype=np.float32).ravel()
                        for i in indice])
    return float(roc_auc_score(m, p))


def _recall(puntajes, indice, umbral):
    if not indice:
        return float("nan")
    s = np.asarray(puntajes, dtype=float)[indice]
    return float((s >= umbral).mean())


def desglose_por_tipo(conjunto_datos, categoria, muestras, puntajes_imagen, etiquetas,
                      mapas_pixeles, mascaras, umbral, con_pixel=True):
    '''Calcula el desglose de R6 de una categoria.

    ret: (fila_categoria | None, filas_por_tipo). Devuelve None en la primera
    posicion cuando la categoria no tiene tipo reservado (`toothbrush`) o el
    dataset no clasifica sus anomalias por tipo (VisA): en esos casos R6 no es
    verificable y no se inventa un numero.
    '''
    reservado = reservado_de(conjunto_datos, categoria)
    tipos = tipos_de_muestras(muestras, conjunto_datos, categoria)
    _, conteo = cargar_manifiesto()

    ind_conformes = [i for i, l in enumerate(etiquetas) if l == 0]
    por_tipo = {}
    for i, (l, t) in enumerate(zip(etiquetas, tipos)):
        if l == 1:
            por_tipo.setdefault(t, []).append(i)
    # Sin anotacion de tipo (VisA) no hay nada que desglosar. Con un unico tipo
    # (`toothbrush`) si se emite su fila, que documenta por que esa categoria
    # queda fuera del contraste, pero no la fila de categoria: no existe
    # reserva posible.
    if not por_tipo or set(por_tipo) == {""}:
        return None, []

    # Una fila por tipo de defecto: situa al reservado en su categoria
    filas_tipo = []
    for t in sorted(por_tipo):
        sub = ind_conformes + por_tipo[t]
        filas_tipo.append({
            "categoria": categoria, "tipo": t,
            "reservado": int(t == reservado),
            "n_train": conteo.get((conjunto_datos, categoria, t, "train"), 0),
            "n_test": len(por_tipo[t]),
            "auroc_img": _auroc_imagen(puntajes_imagen, etiquetas, sub),
            "auroc_pix": (_auroc_pixel(mapas_pixeles, mascaras, sub) if con_pixel
                          else float("nan")),
            "recall": _recall(puntajes_imagen, por_tipo[t], umbral),
        })

    if reservado is None or reservado not in por_tipo:
        return None, filas_tipo

    # Las dos mitades, con las MISMAS conformes
    ind_reservado = por_tipo[reservado]
    ind_vistos = [i for t, v in por_tipo.items() if t != reservado for i in v]
    if not ind_vistos:
        return None, filas_tipo

    ai_v = _auroc_imagen(puntajes_imagen, etiquetas, ind_conformes + ind_vistos)
    ai_r = _auroc_imagen(puntajes_imagen, etiquetas, ind_conformes + ind_reservado)
    ap_v = (_auroc_pixel(mapas_pixeles, mascaras, ind_conformes + ind_vistos) if con_pixel
            else float("nan"))
    ap_r = (_auroc_pixel(mapas_pixeles, mascaras, ind_conformes + ind_reservado) if con_pixel
            else float("nan"))

    # Posicion del tipo reservado entre todos los tipos de la categoria,
    # ordenados de peor a mejor AUROC de imagen. 1 = el peor de todos.
    orden = sorted(filas_tipo,
                   key=lambda f: (np.isnan(f["auroc_img"]), f["auroc_img"]))
    rango = 1 + [f["tipo"] for f in orden].index(reservado)

    fila = {
        "categoria": categoria, "reservado": reservado,
        "n_test_good": len(ind_conformes), "n_vistos": len(ind_vistos),
        "n_reservado": len(ind_reservado),
        "auroc_img_vistos": ai_v, "auroc_img_reservado": ai_r,
        "delta_img": ai_r - ai_v,
        "auroc_pix_vistos": ap_v, "auroc_pix_reservado": ap_r,
        "delta_pix": ap_r - ap_v,
        "recall_vistos": _recall(puntajes_imagen, ind_vistos, umbral),
        "recall_reservado": _recall(puntajes_imagen, ind_reservado, umbral),
        "fpr_good": _recall(puntajes_imagen, ind_conformes, umbral),
        "umbral": float(umbral),
        "rango_reservado": rango, "n_tipos": len(filas_tipo),
    }
    return fila, filas_tipo


# %% persistencia

def guardar_filas(ruta_fichero, campos, filas, clave="categoria"):
    '''Escribe filas reemplazando las que ya existan con la misma clave.

    No se hace `append` a secas porque el pipeline admite `--reanudar` y
    reejecuciones parciales de una categoria: acumular las dos versiones
    dejaria en el mismo fichero numeros de dos ejecuciones distintas, que es
    justo el error que la opcion `--reanudar` documenta como inaceptable.
    '''
    os.makedirs(os.path.dirname(ruta_fichero), exist_ok=True)
    previas = []
    if os.path.exists(ruta_fichero):
        with open(ruta_fichero, newline="", encoding="utf-8") as f:
            previas = list(csv.DictReader(f))
    nuevas = {f[clave] for f in filas}
    previas = [r for r in previas if r.get(clave) not in nuevas]
    with open(ruta_fichero, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(campos))
        w.writeheader()
        for r in previas + [{k: f.get(k, "") for k in campos} for f in filas]:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in r.items()})


def volcar_puntajes(ruta_fichero, muestras, tipos, puntajes_imagen, etiquetas, umbral):
    '''Persiste el score de cada imagen de prueba junto a su ruta y su tipo.

    Sin esto, cualquier replanteamiento posterior del analisis a nivel de
    imagen obligaria a repetir el run completo, porque el MLP de la
    distribucion condicionada no se guarda en disco y supone la mayor parte del
    coste por categoria. Con el fichero, el analisis de imagen se rehace en
    segundos; solo el de pixel seguiria exigiendo GPU.
    '''
    os.makedirs(os.path.dirname(ruta_fichero), exist_ok=True)
    with open(ruta_fichero, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("archivo", "etiqueta", "tipo_defecto", "score", "umbral"))
        for (ruta, _l, _m), t, s, l in zip(muestras, tipos, puntajes_imagen, etiquetas):
            w.writerow((_rel(ruta), l, t, "%.6f" % s, "%.6f" % umbral))


def mapas_calor_reservado(cargador_prueba, mapas_pixeles, mascaras, puntajes_imagen, tipos,
                       reservado, dir_salida, categoria, n=3):
    '''Mapas de calor del tipo NO representado (medio de verificacion de R6).

    Se eligen el caso mejor puntuado, el mediano y el peor, en vez de los tres
    mejores. Publicar solo los aciertos daria una vision sesgada del
    comportamiento ante un tipo no visto; el caso peor puntuado es
    precisamente el que sostiene o refuta la afirmacion de generalizacion.

    Reutiliza los mapas ya calculados: no vuelve a puntuar ninguna imagen.
    '''
    from glob import glob

    from modelo_semisupervisado.evaluacion import guardar_mapa_calor

    indice = [i for i, t in enumerate(tipos) if t == reservado]
    if not indice:
        return []
    os.makedirs(dir_salida, exist_ok=True)
    for viejo in glob(os.path.join(dir_salida, "%s_reservado_*.png" % categoria)):
        os.remove(viejo)

    orden = sorted(indice, key=lambda i: puntajes_imagen[i])
    etiquetas = {orden[-1]: "mejor", orden[0]: "peor"}
    if len(orden) > 2 and n >= 3:
        etiquetas.setdefault(orden[len(orden) // 2], "mediana")

    hechos, ultimo = [], max(etiquetas)
    for i, (imagen, _etiqueta, _mascara) in enumerate(cargador_prueba):
        if i in etiquetas:
            p = os.path.join(dir_salida, "%s_reservado_%s_%s_%d.png"
                             % (categoria, reservado, etiquetas[i], i))
            guardar_mapa_calor(imagen, mapas_pixeles[i], mascaras[i], puntajes_imagen[i], p)
            hechos.append(p)
        if i >= ultimo:
            break                       # no hace falta recorrer el resto
    return hechos


# %% informe agregado

def _leer(ruta_fichero, numericos):
    if not os.path.exists(ruta_fichero):
        return []
    filas = []
    with open(ruta_fichero, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in numericos:
                if k in r:
                    try:
                        r[k] = float(r[k])
                    except (TypeError, ValueError):
                        r[k] = float("nan")
            filas.append(r)
    return filas


def informe(dir_salida, conjunto_datos="mvtec"):
    '''Imprime el informe de R6 y devuelve (filas_categoria, filas_tipo).'''
    d = os.path.join(dir_salida, "generalizacion")
    categorias = _leer(os.path.join(d, "detalle_%s.csv" % conjunto_datos),
                 [c for c in CAMPOS_CAT if c not in ("categoria", "reservado")])
    tipos = _leer(os.path.join(d, "tipos_%s.csv" % conjunto_datos),
                  ["reservado", "n_train", "n_test", "auroc_img", "auroc_pix",
                   "recall"])
    if not categorias:
        print("No hay datos de generalizacion en %s.\n"
              "Ejecuta antes el benchmark:  python -m modelo_semisupervisado.principal --conjunto-datos %s"
              % (d, conjunto_datos))
        return categorias, tipos

    categorias.sort(key=lambda r: r["categoria"])
    print("=" * 100)
    print("R6 - GENERALIZACION A TIPOS DE DEFECTO NO REPRESENTADOS  "
          "(%s, %d categorias)" % (conjunto_datos, len(categorias)))
    print("=" * 100)
    print("Criterio (IOV): detectar al menos el %.0f %% de las anomalias del tipo "
          "no representado,\nen el punto de operacion declarado para la categoria "
          "(umbral calibrado con las dos clases\nde ENTRENAMIENTO, nunca con el "
          "conjunto de prueba).\n" % (100 * RECALL_OBJETIVO))

    cab = ("%-12s %-16s %5s %8s %8s %8s   %8s %8s   %6s" %
           ("categoria", "tipo reservado", "n", "AUROCv", "AUROCr", "delta",
            "recall_v", "recall_r", "R6"))
    print(cab)
    print("-" * len(cab))
    cumplen = 0
    for r in categorias:
        ok = r["recall_reservado"] >= RECALL_OBJETIVO
        cumplen += int(ok)
        print("%-12s %-16s %5d %8.4f %8.4f %+8.4f   %8.3f %8.3f   %6s" %
              (r["categoria"], r["reservado"], int(r["n_reservado"]),
               r["auroc_img_vistos"], r["auroc_img_reservado"], r["delta_img"],
               r["recall_vistos"], r["recall_reservado"], "SI" if ok else "NO"))

    n = len(categorias)
    det = sum(r["recall_reservado"] * r["n_reservado"] for r in categorias)
    tot = sum(r["n_reservado"] for r in categorias)

    def m(k):
        return sum(r[k] for r in categorias) / n

    print("-" * len(cab))
    print("%-12s %-16s %5d %8.4f %8.4f %+8.4f   %8.3f %8.3f   %2d/%d" %
          ("PROMEDIO", "", tot, m("auroc_img_vistos"), m("auroc_img_reservado"),
           m("delta_img"), m("recall_vistos"), m("recall_reservado"), cumplen, n))
    print("\nRecall agregado sobre las %d anomalias reservadas de las %d "
          "categorias: %.4f (%d detectadas)"
          % (tot, n, det / max(1, tot), int(round(det))))
    print("AUROC de pixel: vistos %.4f | reservado %.4f | delta %+.4f"
          % (m("auroc_pix_vistos"), m("auroc_pix_reservado"), m("delta_pix")))
    print("Tasa de falsa alarma sobre las conformes de prueba, en el mismo punto "
          "de operacion: %.4f" % m("fpr_good"))

    # Contraste pareado. El AUROC de la mitad vista y el de la reservada se
    # miden sobre la MISMA categoria y comparten las conformes, asi que estan
    # pareados y procede un contraste de rangos con signo (Demsar 2006), como
    # en el resto de comparaciones del proyecto. Con n = 11 la prueba tiene
    # poca potencia: un resultado no significativo NO demuestra igualdad.
    try:
        from scipy.stats import wilcoxon
        dif = [r["delta_img"] for r in categorias]
        if any(abs(x) > 0 for x in dif):
            _st, p = wilcoxon([r["auroc_img_vistos"] for r in categorias],
                              [r["auroc_img_reservado"] for r in categorias])
            peor = sum(1 for x in dif if x < 0)
            print("\nWilcoxon pareado (AUROC imagen, vistos vs reservado): "
                  "p=%.4f | el reservado rinde peor en %d de %d categorias"
                  % (p, peor, n))
            print("  n=%d: potencia baja. Un p alto no prueba que no haya "
                  "diferencia, solo que no se detecta." % n)
    except ImportError:
        pass

    # Situar al tipo reservado dentro de la dificultad de su propia categoria.
    if tipos:
        print("\n--- El tipo reservado, es peor por no visto o por dificil? ---")
        ultimos = sum(1 for r in categorias if r["rango_reservado"] == 1)
        # Posicion normalizada al intervalo [0, 1]: 0 cuando el reservado es el
        # peor tipo de su categoria y 1 cuando es el mejor. Se normaliza por
        # (n_tipos - 1) y no por n_tipos para que las categorias con distinto
        # numero de tipos sean promediables entre si.
        medio = sum((r["rango_reservado"] - 1) / max(1, r["n_tipos"] - 1)
                    for r in categorias) / n
        print("El reservado es el PEOR tipo de su categoria en %d de %d casos."
              % (ultimos, n))
        print("Posicion relativa media (0 = siempre el peor, 1 = siempre el "
              "mejor): %.2f" % medio)
        print("Bajo la hipotesis de que no ser visto no influye, la posicion "
              "esperada seria 0.50.")
        print("Se reporta el AUROC de CADA tipo en tipos_%s.csv: el reservado "
              "debe leerse dentro\nde ese rango, no contra el promedio de su "
              "categoria." % conjunto_datos)

    faltan = [r["categoria"] for r in categorias if r["n_reservado"] == 0]
    if faltan:
        print("\n[AVISO] Sin anomalas reservadas en: %s" % ", ".join(faltan))
    print("\nVeredicto R6: %s (%d de %d categorias alcanzan el %.0f %% de recall "
          "sobre el tipo no representado)"
          % ("CUMPLIDO" if cumplen == n else "CUMPLIDO PARCIALMENTE",
             cumplen, n, 100 * RECALL_OBJETIVO))
    return categorias, tipos


if __name__ == "__main__":
    from modelo_semisupervisado.configuracion import obtener_configuracion
    informe(obtener_configuracion().dir_salida, "mvtec")

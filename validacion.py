'''
Validacion de los resultados propios contra la literatura del paradigma (R3).

QUE HACE ESTE MODULO. El resultado R3 exige contrastar las cifras obtenidas con
las publicadas por los metodos del paradigma seleccionado en el R1 -banco de
memoria con caracteristicas preentrenadas y comparacion directa-, que son dos:

    PNI        Bae et al., ICCV 2023            referencia principal
    SoftPatch+ Wang et al., Pattern Recognition 2025

Lee los CSV producidos por `modelo_semisupervisado.principal` y los contrasta categoria a
categoria contra las tablas transcritas en `modelo_semisupervisado.referencias`, que a su vez
estan verificadas contra los promedios que imprimen los articulos.

POR QUE SE RECALCULAN LOS PROMEDIOS PUBLICADOS. La delimitacion del alcance deja
doce categorias de MVTec AD y cinco de VisA. Comparar un promedio de doce contra
un promedio publicado de quince no seria valido: bastaria con que las excluidas
fueran las dificiles para que la diferencia se debiera al reparto y no al
metodo. Por eso se promedian las tablas desglosadas sobre EL MISMO subconjunto.

Ejecutar:  .\\venv\\Scripts\\python.exe -m modelo_semisupervisado.validacion
'''
import csv
import os

from modelo_semisupervisado import referencias as R

RAIZ = os.path.dirname(os.path.abspath(__file__))
SALIDAS = os.path.join(RAIZ, "outputs")

# Margen del indicador del R3: igualar la cifra publicada con una tolerancia de
# un punto porcentual. No es "superar": se fijo asi en el R1 y no se cambia
# despues de ver los resultados.
MARGEN = 1.0


# %% Lectura de resultados propios

def leer(nombre):
    '''Lee un resultados_*.csv y devuelve {categoria: (imagen, pixel)} en %.'''
    ruta = os.path.join(SALIDAS, nombre)
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            "falta %s: ejecuta antes `python -m modelo_semisupervisado.principal --conjunto-datos %s`"
            % (ruta, nombre.split("_")[1].split(".")[0]))
    salida = {}
    for fila in csv.DictReader(open(ruta, encoding="utf-8")):
        if fila["category"] == "PROMEDIO":
            continue
        salida[fila["category"]] = (float(fila["auroc_image"]) * 100.0,
                                 float(fila["auroc_pixel"]) * 100.0)
    return salida


def _veredicto(obtenido, referencia):
    '''IGUALADO si la diferencia cae dentro del margen del R3, o si se supera.'''
    d = obtenido - referencia
    if d >= 0:
        return d, "SUPERA"
    return d, ("IGUALADO" if -d <= MARGEN else "POR DEBAJO")


def _linea(etiqueta, obtenido, referencia, ancho=13):
    d, v = _veredicto(obtenido, referencia)
    return "  %-*s %7.2f  vs %7.2f   %+6.2f   %s" % (
        ancho, etiqueta, obtenido, referencia, d, v)


def _cabecera(texto):
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


# %% Contraste categoria a categoria

def contra_pni_mvtec(nuestro):
    '''MVTec AD por categoria contra la Tabla 1 de PNI.

    La Tabla 1 de PNI corresponde al metodo COMPLETO, con la red de refinamiento
    R, que esta implementacion no incluye (se documento como fase opcional en el
    R1 porque aporta +0.10/+0.18 a cambio de entrenar un DenseNet161 supervisado
    sobre anomalias sinteticas). Se reportan por eso DOS varas: la publicada tal
    cual, y la corregida restando el efecto que la Tabla 3 atribuye a R. La
    segunda es una ESTIMACION -el articulo no desglosa la variante N+P por
    categoria- y solo se aplica al promedio, nunca a una categoria suelta.
    '''
    _cabecera("MVTec AD  |  PNI (Bae et al., ICCV 2023), Tabla 1 (metodo completo)")
    categorias = [c for c in R.MVTEC_ALCANCE if c in nuestro]
    print("  %-12s %-21s %-21s" % ("", "AUROC imagen", "AUROC pixel"))
    print("  %-12s %7s %7s %6s  %7s %7s %6s"
          % ("categoria", "propio", "PNI", "delta", "propio", "PNI", "delta"))
    print("  " + "-" * 62)
    for c in categorias:
        o, r = nuestro[c], R.PNI_MVTEC[c]
        print("  %-12s %7.2f %7.2f %+6.2f  %7.2f %7.2f %+6.2f"
              % (c, o[0], r[0], o[0] - r[0], o[1], r[1], o[1] - r[1]))

    m = R.referencia_mvtec()
    oi = sum(nuestro[c][0] for c in categorias) / len(categorias)
    op = sum(nuestro[c][1] for c in categorias) / len(categorias)
    print()
    print("  Promedios recalculados sobre las %d categorias del alcance:" % len(categorias))
    print(_linea("imagen", oi, m["completo"][0]))
    print(_linea("pixel", op, m["completo"][1]))
    print()
    print("  Contra la variante N+P, sin la red de refinamiento R (APROXIMADA,")
    print("  se resta el efecto que la Tabla 3 atribuye a R: %+.2f img / %+.2f pix):"
          % (-m["delta_R"][0], -m["delta_R"][1]))
    print(_linea("imagen*", oi, m["sin_R"][0]))
    print(_linea("pixel*", op, m["sin_R"][1]))
    return (oi, op)


def contra_pni_visa(nuestro):
    '''VisA contra la Tabla 6 de PNI, que agrupa por dominio y no por categoria.

    De las cinco categorias del alcance, las cuatro placas de circuito impreso
    son EXACTAMENTE el dominio de estructura compleja, asi que su cifra publicada
    se aplica sin transformacion. `candle` pertenece a un dominio del que las
    otras tres quedan excluidas y de una media de cuatro valores no se puede
    extraer uno: se reporta sin referencia. Asignarle el promedio de su dominio
    (87.9 de imagen) daria una meta artificialmente baja, sesgada a favor propio.
    '''
    _cabecera("VisA  |  PNI (Bae et al., ICCV 2023), Tabla 6 (agrupada por dominio)")
    pcb = [c for c in ("pcb1", "pcb2", "pcb3", "pcb4") if c in nuestro]
    ref = R.PNI_VISA_DOMINIO["estructura compleja"]
    oi = sum(nuestro[c][0] for c in pcb) / len(pcb)
    op = sum(nuestro[c][1] for c in pcb) / len(pcb)
    print("  Dominio de estructura compleja = pcb1..pcb4, las cuatro en alcance.")
    print("  La referencia se aplica SIN transformacion: el dominio publicado y")
    print("  nuestro subconjunto son el mismo conjunto de categorias.")
    print()
    for c in pcb:
        print("  %-12s %7.2f img  %7.2f pix" % (c, nuestro[c][0], nuestro[c][1]))
    print("  " + "-" * 40)
    print(_linea("imagen", oi, ref[0]))
    print(_linea("pixel", op, ref[1]))
    if "candle" in nuestro:
        print()
        print("  candle: %.2f img / %.2f pix  -- SIN referencia individual publicada."
              % nuestro["candle"])
        print("  (su dominio, multiples instancias, pierde por alcance las otras")
        print("   tres categorias; el promedio del dominio no le es atribuible)")
    return (oi, op)


def contra_softpatch(nuestro, tabla_img, tabla_pix, alcance, titulo, ruido):
    '''Contraste contra SoftPatch+, que mide otra cosa. Leer la advertencia.

    SoftPatch+ estudia la ROBUSTEZ AL RUIDO DE ETIQUETA: inyecta anomalias en el
    conjunto de entrenamiento SIN etiquetar, las trata como normales y su metodo
    trata de anular su efecto. Aqui las anomalias del entrenamiento estan
    ETIQUETADAS y se explotan. Sobre la misma proporcion nominal (10 % en MVTec,
    8 % en VisA) los dos trabajos persiguen objetivos OPUESTOS, de modo que estas
    cifras SITUAN el resultado pero no constituyen una meta ni un veredicto del
    R3. Se reportan porque es el segundo estudio del paradigma y porque su
    desglose por categoria es el unico publicado para VisA.
    '''
    _cabecera(titulo)
    print("  ADVERTENCIA DE PROTOCOLO: SoftPatch+ entrena con un %s de anomalias" % ruido)
    print("  NO ETIQUETADAS (ruido que trata de anular); aqui son ETIQUETADAS y se")
    print("  explotan. Ademas usa 256 -> 224 px frente a nuestros 512 -> 480, y su")
    print("  modalidad 'No Overlap' RETIRA del test las anomalas inyectadas.")
    print("  No es un veredicto del R3: es contexto.")
    print()
    categorias = [c for c in alcance if c in nuestro]
    print("  %-12s %-23s %-23s" % ("", "AUROC imagen", "AUROC pixel"))
    print("  %-12s %7s %7s %7s  %7s %7s %7s"
          % ("categoria", "propio", "SP+ NO", "SP+ Ov", "propio", "SP+ NO", "SP+ Ov"))
    print("  " + "-" * 66)
    for c in categorias:
        o, ri, rp = nuestro[c], tabla_img[c], tabla_pix[c]
        print("  %-12s %7.2f %7.2f %7.2f  %7.2f %7.2f %7.2f"
              % (c, o[0], ri[0], ri[1], o[1], rp[0], rp[1]))
    print("  " + "-" * 66)
    oi = sum(nuestro[c][0] for c in categorias) / len(categorias)
    op = sum(nuestro[c][1] for c in categorias) / len(categorias)
    print("  %-12s %7.2f %7.2f %7.2f  %7.2f %7.2f %7.2f"
          % ("PROMEDIO", oi,
             R.promedio(tabla_img, categorias, indice=0), R.promedio(tabla_img, categorias, indice=1),
             op,
             R.promedio(tabla_pix, categorias, indice=0), R.promedio(tabla_pix, categorias, indice=1)))
    print("  (SP+ NO = No Overlap, SP+ Ov = Overlap; promedios recalculados sobre")
    print("   las mismas %d categorias)" % len(categorias))


# %% Informe por consola

def _informe():
    fallos = R.verificar_transcripcion()
    _cabecera("VERIFICACION DE LA TRANSCRIPCION DE LAS TABLAS PUBLICADAS")
    if fallos:
        for f in fallos:
            print("  FALLO:", f)
        print("  -> hay un valor mal copiado; el contraste NO es fiable")
        return
    print("  Los promedios calculados a partir de los valores por categoria")
    print("  reproducen los que imprimen los articulos. Transcripcion OK.")

    mv = leer("resultados_mvtec.csv")
    vs = leer("resultados_visa.csv")

    contra_pni_mvtec(mv)
    contra_softpatch(mv, R.SOFTPATCH_MVTEC_IMAGEN, R.SOFTPATCH_MVTEC_PIXEL,
                     R.MVTEC_ALCANCE,
                     "MVTec AD  |  SoftPatch+ (Wang et al., PR 2025), Tablas 1 y 2",
                     "10 %")
    contra_pni_visa(vs)
    contra_softpatch(vs, R.SOFTPATCH_VISA_IMAGEN, R.SOFTPATCH_VISA_PIXEL,
                     R.VISA_ALCANCE,
                     "VisA  |  SoftPatch+ (Wang et al., PR 2025), Tablas 3 y 4",
                     "8 %")

    _cabecera("VEREDICTO DEL R3 (referencia principal: PNI)")
    m = R.referencia_mvtec()
    categorias = [c for c in R.MVTEC_ALCANCE if c in mv]
    oi = sum(mv[c][0] for c in categorias) / len(categorias)
    op = sum(mv[c][1] for c in categorias) / len(categorias)
    pcb = [c for c in ("pcb1", "pcb2", "pcb3", "pcb4") if c in vs]
    vi = sum(vs[c][0] for c in pcb) / len(pcb)
    vp = sum(vs[c][1] for c in pcb) / len(pcb)
    ref_v = R.PNI_VISA_DOMINIO["estructura compleja"]
    print("  Criterio: igualar la cifra publicada con un margen de %.0f punto"
          % MARGEN)
    print("  porcentual, fijado a priori en el R1.")
    print()
    print(_linea("MVTec imagen", oi, m["sin_R"][0], ancho=14))
    print(_linea("MVTec pixel", op, m["sin_R"][1], ancho=14))
    print(_linea("VisA imagen", vi, ref_v[0], ancho=14))
    print(_linea("VisA pixel", vp, ref_v[1], ancho=14))
    print()
    print("  MVTec se contrasta contra la variante N+P estimada (sin la red de")
    print("  refinamiento); VisA, contra el dominio de estructura compleja, que")
    print("  es exacto. `candle` queda sin referencia publicada.")


if __name__ == "__main__":
    _informe()

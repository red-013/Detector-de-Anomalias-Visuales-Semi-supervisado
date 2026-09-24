'''
Modulo (f): orquestador del pipeline completo + logging.

Recorre las categorias pedidas y, para cada una:
    loaders (a) -> extractor (b) -> banco+coreset (c) -> scoring (d) -> metricas (e)
Registra en log: tiempos, tamano del banco y metricas por categoria; guarda el
banco en disco, genera heatmaps y escribe un CSV con la fila promedio.

Uso (desde C:\\Users\\USER\\Desktop\\tesis):
    python -m modelo_semisupervisado.principal                      # las 15 categorias
    python -m modelo_semisupervisado.principal --categorias bottle  # una sola
    python -m modelo_semisupervisado.principal --categorias bottle,cable,grid
    python -m modelo_semisupervisado.principal --sin-mapas-calor
'''
import argparse
import csv
import logging
import os
import time

import torch

from modelo_semisupervisado.configuracion import (obtener_configuracion, fijar_semilla, categorias_de,
                              resolver_rutas)
from modelo_semisupervisado.conjuntos_datos import construir_cargadores, construir_cargadores_particion
from modelo_semisupervisado.semisupervisado import ajustar_semisupervisado, umbral_balanceado
from modelo_semisupervisado.extractor_caracteristicas import ExtractorJerarquico
from modelo_semisupervisado.banco_memoria import BancoDeMemoria
from modelo_semisupervisado.distribucion import DistribucionCondicional
from modelo_semisupervisado.puntuacion import Evaluador
from modelo_semisupervisado.evaluacion import evaluar, generar_mapas_calor
from modelo_semisupervisado import generalizacion as G


# %% Cerrojo de ejecucion

def adquirir_cerrojo(conf, comp, registro):
    '''Impide que dos runs del mismo dataset corran a la vez.

    Motivo real (2026-08-24): en Windows, matar el proceso que LANZO el run no
    mata al interprete de Python. Un run que se daba por terminado siguio vivo,
    y al arrancar otro encima los dos escribieron en el mismo run.log y en el
    mismo CSV parcial, ademas de pelearse por la GPU (una categoria paso de
    540 s a 1460 s). El cerrojo guarda el PID: si el proceso que lo creo ya no
    existe, se considera obsoleto y se reutiliza, asi que una caida no deja el
    pipeline bloqueado.
    '''
    ruta = os.path.join(conf.dir_salida, f".lock_{conf.conjunto_datos}_{comp}")
    if os.path.exists(ruta):
        try:
            viejo = int(open(ruta, encoding="utf-8").read().split()[0])
        except (ValueError, IndexError):
            viejo = None
        if viejo is not None and _pid_vivo(viejo):
            registro.error(
                f"Ya hay un run de '{conf.conjunto_datos}' [{comp}] en marcha "
                f"(PID {viejo}). Dos procesos a la vez corromperian el CSV "
                f"parcial y se pelearian por la GPU. Espera a que termine, o "
                f"detenlo y borra {ruta}.")
            return None
        registro.warning(f"Cerrojo obsoleto de un PID muerto ({viejo}): se reutiliza.")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(f"{os.getpid()}\n")
    return ruta


def _pid_vivo(pid):
    '''True si el PID existe. Sin dependencias externas (no hay psutil).'''
    if os.name == "nt":
        import subprocess
        salida = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in salida
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


# %% Registro de ejecucion

def configurar_registro(dir_salida):
    registro = logging.getLogger("modelo_semisupervisado")
    registro.setLevel(logging.INFO)
    registro.handlers.clear()
    formato = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    manejador_fichero = logging.FileHandler(os.path.join(dir_salida, "run.log"), encoding="utf-8")
    manejador_fichero.setFormatter(formato)
    registro.addHandler(manejador_fichero)
    manejador_consola = logging.StreamHandler()
    manejador_consola.setFormatter(formato)
    registro.addHandler(manejador_consola)
    return registro


# %% Evaluacion de una categoria

def puntuar_validacion(evaluador, extractor, cargador):
    '''Scores de imagen de un cargador de entrenamiento.

    Sirve tanto para las conformes de validacion (que devuelven solo la imagen)
    como para las anomalas etiquetadas (que devuelven la terna imagen, etiqueta,
    mascara), de modo que ambas clases se puntuan con el MISMO codigo y el
    umbral calibrado no dependa de por que rama se haya pasado.
    '''
    puntajes = []
    for lote in cargador:
        imagenes = lote[0] if isinstance(lote, (list, tuple)) else lote
        for i in range(imagenes.shape[0]):
            s, _ = evaluador.puntuar(extractor, imagenes[i:i + 1])
            puntajes.append(s)
    return puntajes


def analizar_generalizacion(conf, categoria, cargador_prueba, puntajes_imagen, etiquetas,
                            mapas_pixeles, mascaras, umbral, registro,
                            con_mapas_calor=True):
    '''Desglose por tipo de defecto del conjunto de prueba (resultado R6).

    Se calcula AQUI y no en un modulo posterior por una razon de coste: el
    desglose a nivel de pixel necesita los mapas de anomalia, que no se
    persisten -pesan 1,8 MB por imagen- y cuya regeneracion exigiria repetir el
    entrenamiento del MLP de la distribucion condicionada, que es la mayor
    parte de los ~10 minutos por categoria. Los scores por imagen SI se vuelcan
    a disco, de modo que cualquier reconsideracion posterior del analisis a
    nivel de imagen se rehace en segundos y sin GPU.

    Solo se aplica a MVTec AD sobre la particion de R4: es la unica combinacion
    en la que existe un tipo de defecto reservado. Con `--disposicion original` no
    hay reserva alguna, y VisA no clasifica sus anomalias por tipo.
    '''
    conj = getattr(cargador_prueba.dataset, "muestras", None)
    if conj is None or conf.disposicion != "split":
        return None
    tipos = G.tipos_de_muestras(conj, conf.conjunto_datos, categoria)
    G.volcar_puntajes(os.path.join(conf.dir_salida, "scores",
                                 f"{conf.conjunto_datos}_{categoria}.csv"),
                    conj, tipos, puntajes_imagen, etiquetas, umbral)
    if conf.conjunto_datos != "mvtec":
        return None

    fila, filas_tipo = G.desglose_por_tipo(conf.conjunto_datos, categoria, conj,
                                           puntajes_imagen, etiquetas, mapas_pixeles,
                                           mascaras, umbral)
    destino = os.path.join(conf.dir_salida, "generalizacion")
    if filas_tipo:
        G.guardar_filas(os.path.join(destino, f"tipos_{conf.conjunto_datos}.csv"),
                        G.CAMPOS_TIPO, filas_tipo)
    if fila is None:
        registro.info("  [R6] %s sin tipo reservado: queda fuera del analisis de "
                    "generalizacion (%d tipo(s) de defecto en prueba)",
                    categoria, len(filas_tipo))
        return None

    G.guardar_filas(os.path.join(destino, f"detalle_{conf.conjunto_datos}.csv"),
                    G.CAMPOS_CAT, [fila])
    if con_mapas_calor:
        G.mapas_calor_reservado(cargador_prueba, mapas_pixeles, mascaras, puntajes_imagen,
                             tipos, fila["reservado"],
                             os.path.join(destino, "heatmaps", conf.conjunto_datos),
                             categoria, n=conf.n_mapas_calor)
    registro.info("  [R6] tipo reservado '%s' (%d img): AUROC img %.4f vs %.4f de "
                "los tipos vistos (%+.4f) | recall %.3f vs %.3f | %s el %.0f %%",
                fila["reservado"], fila["n_reservado"],
                fila["auroc_img_reservado"], fila["auroc_img_vistos"],
                fila["delta_img"], fila["recall_reservado"],
                fila["recall_vistos"],
                "CUMPLE" if fila["recall_reservado"] >= G.RECALL_OBJETIVO
                else "NO CUMPLE", 100 * G.RECALL_OBJETIVO)
    return fila


def ejecutar_categoria(conf, categoria, extractor, registro, con_mapas_calor=True,
                 guardar_banco=True):
    t_cat = time.time()
    # La particion de R4 (`data/datasets/tesis_split/`) es la fuente por defecto: fija
    # que imagenes ve el modelo y cuales quedan reservadas, y es la unica que
    # expone las anomalas etiquetadas. `--disposicion original` vuelve a leer los
    # corpus sin repartir, para reproducir los resultados anteriores.
    cargador_anomalas = None
    if conf.disposicion == 'split':
        (cargador_entrenamiento, cargador_validacion, cargador_anomalas,
         cargador_prueba) = construir_cargadores_particion(conf, categoria)
    else:
        (cargador_entrenamiento, cargador_validacion,
         cargador_prueba) = construir_cargadores(conf, categoria)

    # (c) banco: C_emb + C_dist + mapeo
    banco = BancoDeMemoria(conf)
    st = banco.ajustar(extractor, cargador_entrenamiento)
    if guardar_banco:      # en ablations se desactiva: 10 MB x 15 cats x N variantes
        banco.guardar(os.path.join(conf.dir_salida,
                               f"bank_{conf.conjunto_datos}_{categoria}.pt"))
    registro.info(
        f"[{categoria}] parches {st['n_full']} (pool {st['n_pool']}) "
        f"-> C_emb {st['n_coreset']} -> C_dist {st['n_dist']} "
        f"(dim {st['dim']}, grid {st['grid']}) | "
        f"extract {st['t_extract_s']}s coreset {st['t_coreset_s']}s "
        f"dist {st['t_dist_s']}s"
    )

    # (c-ter) regimen semisupervisado: C_neg + depuracion de C_emb.
    # Va DESPUES del banco y ANTES de la distribucion: la purga modifica C_emb
    # y el mapeo emb2dist que usa el MLP debe rehacerse sobre el banco final.
    banco_neg, st_semi = ajustar_semisupervisado(extractor, cargador_anomalas, banco, conf, registro)

    # (c-bis) distribucion condicionada a posicion y vecindad
    dist = None
    if conf.usar_vecindad or conf.usar_posicion:
        t_d = time.time()
        dist = DistribucionCondicional(conf, banco).ajustar(extractor, cargador_entrenamiento, registro)
        comp = "+".join([c for c, activo in (("N", conf.usar_vecindad),
                                         ("P", conf.usar_posicion)) if activo])
        registro.info(f"[{categoria}] distribucion {comp} ajustada en "
                    f"{time.time()-t_d:.1f}s ({st['n_dist']} simbolos)")

    # (d) scoring
    evaluador = Evaluador(banco, conf, distribucion=dist, banco_neg=banco_neg)
    t0 = time.time()
    puntajes_validacion = puntuar_validacion(evaluador, extractor, cargador_validacion)
    puntajes_imagen, etiquetas, mapas_pixeles, mascaras = evaluador.puntuar_cargador(
        extractor, cargador_prueba)
    t_inferencia = time.time() - t0

    # Umbral de operacion. Con anomalas etiquetadas DE ENTRENAMIENTO el punto
    # de corte se calibra con las dos clases sin tocar el test; sin ellas se
    # mantiene el percentil de las conformes de validacion.
    umbral_semi, f1_calib = None, float('nan')
    if conf.regimen == 'semi' and cargador_anomalas is not None and conf.modo_umbral == 'balanced':
        puntajes_anomalas = puntuar_validacion(evaluador, extractor, cargador_anomalas)
        umbral_semi, f1_calib = umbral_balanceado(puntajes_validacion, puntajes_anomalas)
        registro.info("  [semi] umbral calibrado con 2 clases: %.4f (F1 de calibracion %.3f)",
                    umbral_semi, f1_calib)

    # (e) metricas
    res = evaluar(puntajes_imagen, etiquetas, mapas_pixeles, mascaras, puntajes_validacion,
                   metodo=conf.metodo_honesto, k=conf.k_honesto,
                   percentil=conf.percentil_honesto,
                   umbral_forzado=umbral_semi)
    if con_mapas_calor:
        generar_mapas_calor(evaluador, extractor, cargador_prueba, puntajes_imagen, etiquetas,
                          os.path.join(conf.dir_salida, "heatmaps", conf.conjunto_datos),
                          categoria, n=conf.n_mapas_calor)

    # (g) generalizacion a tipos de defecto no representados (R6). Usa el mismo
    # umbral con el que se reporta la categoria: el punto de operacion no se
    # reelige para este analisis.
    analizar_generalizacion(conf, categoria, cargador_prueba, puntajes_imagen, etiquetas,
                            mapas_pixeles, mascaras, res["threshold_honest"], registro,
                            con_mapas_calor=con_mapas_calor)

    mf1, hon = res["metrics_bestF1"], res["metrics_honest"]
    registro.info(
        f"[{categoria}] AUROC img={res['auroc_image']:.4f} pix={res['auroc_pixel']:.4f} | "
        f"bestF1={mf1['f1']:.3f} (P={mf1['precision']:.3f} R={mf1['recall']:.3f}) | "
        f"honestoF1={hon['f1']:.3f} (P={hon['precision']:.3f} R={hon['recall']:.3f}) | "
        f"infer {t_inferencia:.1f}s total {time.time()-t_cat:.1f}s"
    )
    return {
        "category": categoria,
        "auroc_image": res["auroc_image"],
        "auroc_pixel": res["auroc_pixel"],
        "bestF1_f1": mf1["f1"], "bestF1_precision": mf1["precision"],
        "bestF1_recall": mf1["recall"],
        "honest_f1": hon["f1"], "honest_precision": hon["precision"],
        "honest_recall": hon["recall"],
        "n_val": res["n_val"],
        "n_coreset": st["n_coreset"],
        "n_neg": st_semi["n_neg"],
        "n_purged": st_semi["n_purged"],
    }


# %% Agregacion y contraste con la literatura

def fila_promedio(filas):
    claves = ["auroc_image", "auroc_pixel", "bestF1_f1", "bestF1_precision",
            "bestF1_recall", "honest_f1", "honest_precision", "honest_recall"]
    prom = {"category": "PROMEDIO", "n_val": "", "n_coreset": "",
           "n_neg": "", "n_purged": ""}
    for k in claves:
        prom[k] = sum(r[k] for r in filas) / len(filas)
    return prom


def comparar_con_referencia(conf, filas, comp, registro):
    '''Contrasta el promedio obtenido con las cifras publicadas RECALCULADAS.

    Solo se contrasta contra el subconjunto de categorias para el que existe
    una referencia publicada equiparable. En VisA eso significa que el contraste
    se hace sobre pcb1-4, porque PNI publica ese corpus agrupado por dominio y
    el dominio de estructura compleja coincide exactamente con esas cuatro
    categorias; `candle` se reporta sin referencia, dado que su dominio incluye
    tres categorias excluidas por alcance y de una media de cuatro valores no
    puede despejarse uno.
    '''
    from modelo_semisupervisado import referencias as R

    por_cat = {r["category"]: r for r in filas if r["category"] != "PROMEDIO"}

    def linea(nombre, obt, refv, exacta=True):
        delta = obt - refv
        estado = "IGUALADO" if delta >= -0.01 else "POR DEBAJO"
        marca = "" if exacta else "  (referencia APROXIMADA)"
        registro.info("AUROC %-6s: %.5f | referencia %.4f | delta %+.4f -> %s%s",
                    nombre, obt, refv, delta, estado, marca)

    if conf.conjunto_datos == "mvtec":
        evaluadas = [c for c in R.MVTEC_ALCANCE if c in por_cat]
        if not evaluadas:
            return
        imagen = sum(por_cat[c]["auroc_image"] for c in evaluadas) / len(evaluadas)
        pix = sum(por_cat[c]["auroc_pixel"] for c in evaluadas) / len(evaluadas)
        ref_c = R.promedio(R.PNI_MVTEC, evaluadas)
        dR = (R.PNI_ABLACION["N+P+R"][0] - R.PNI_ABLACION["N+P"][0],
              R.PNI_ABLACION["N+P+R"][1] - R.PNI_ABLACION["N+P"][1])
        registro.info("--- contraste sobre %d categorias de MVTec AD ---", len(evaluadas))
        # La Tabla 1 de PNI corresponde al metodo COMPLETO, con red de
        # refinamiento. Nuestra implementacion no la incluye, asi que se
        # reportan las dos varas: la publicada y la corregida por el efecto que
        # la Tabla 3 atribuye a R. La segunda es una estimacion, porque el
        # articulo no desglosa la variante N+P por categoria.
        linea("imagen", imagen, ref_c[0] / 100.0)
        linea("pixel", pix, ref_c[1] / 100.0)
        linea("imagen*", imagen, (ref_c[0] - dR[0]) / 100.0, exacta=False)
        linea("pixel*", pix, (ref_c[1] - dR[1]) / 100.0, exacta=False)
        registro.info("  * = referencia sin la red de refinamiento R, que esta "
                    "implementacion no incluye")
    else:
        pcb = [c for c in ("pcb1", "pcb2", "pcb3", "pcb4") if c in por_cat]
        if pcb:
            imagen = sum(por_cat[c]["auroc_image"] for c in pcb) / len(pcb)
            pix = sum(por_cat[c]["auroc_pixel"] for c in pcb) / len(pcb)
            r = R.PNI_VISA_DOMINIO["estructura compleja"]
            registro.info("--- contraste sobre %d categorias de estructura "
                        "compleja de VisA ---", len(pcb))
            linea("imagen", imagen, r[0] / 100.0)
            linea("pixel", pix, r[1] / 100.0)
        for c in ("candle",):
            if c in por_cat:
                registro.info("%s: img=%.5f pix=%.5f | SIN referencia individual "
                            "publicada (PNI agrupa VisA por dominio y las otras "
                            "tres categorias de su dominio quedan fuera del alcance)",
                            c, por_cat[c]["auroc_image"], por_cat[c]["auroc_pixel"])
        sp = [c for c in R.VISA_ALCANCE if c in por_cat]
        if len(sp) == len(R.VISA_ALCANCE):
            pix5 = sum(por_cat[c]["auroc_pixel"] for c in sp) / len(sp)
            ref_sp = R.promedio(R.SOFTPATCH_VISA_PIXEL, sp, indice=0)
            registro.info("referencia secundaria SoftPatch+ (pixel, 8 %% de ruido, "
                        "protocolo distinto): obtenido %.5f vs %.4f",
                        pix5, ref_sp / 100.0)


# %% Punto de entrada

def principal():
    analizador = argparse.ArgumentParser()
    analizador.add_argument("--categorias", type=str, default="all",
                        help="'all' o lista separada por comas (bottle,cable,...)")
    analizador.add_argument("--conjunto-datos", type=str, default=None,
                        choices=["mvtec", "visa"],
                        help="dataset a evaluar (por defecto el de configuracion.py)")
    analizador.add_argument("--sin-mapas-calor", action="store_true")
    # Ablation de los componentes de PNI (Tabla 3 del paper):
    #   --sin-vecindad --sin-posicion  ->  baseline sin N ni P
    #   --sin-posicion                    ->  solo N
    #   (por defecto)                    ->  N + P
    analizador.add_argument("--sin-vecindad", action="store_true")
    analizador.add_argument("--sin-posicion", action="store_true")
    analizador.add_argument("--modo-puntaje", type=str, default=None,
                        choices=["max", "sum"],
                        help="agregacion de candidatos: eq.4 (max) o eq.3 (sum)")
    analizador.add_argument("--regimen", type=str, default=None,
                        choices=("oneclass", "semi"),
                        help="regimen de supervision (por defecto: el de config)")
    analizador.add_argument("--disposicion", type=str, default=None,
                        choices=("split", "original"),
                        help="split = particion de R4; original = corpus sin repartir")
    analizador.add_argument("--beta-neg", type=float, default=None,
                        help="peso del termino contrastivo (0 lo desactiva)")
    analizador.add_argument("--razon-purga", type=float, default=None,
                        help="fraccion maxima de C_emb a depurar (0 lo desactiva)")
    analizador.add_argument("--reanudar", action="store_true",
                        help="reaprovecha el CSV parcial: salta las categorias "
                             "ya evaluadas con esta misma combinacion de "
                             "dataset y componentes. Un run de 27 categorias "
                             "dura horas y cualquier corte obligaba a repetirlo "
                             "entero. NO se activa por defecto: si se cambia un "
                             "hiperparametro, reutilizar filas viejas mezclaria "
                             "dos configuraciones en una misma tabla.")
    argumentos = analizador.parse_args()

    conf = obtener_configuracion()
    if argumentos.conjunto_datos:
        conf.conjunto_datos = argumentos.conjunto_datos
        resolver_rutas(conf)          # la raiz depende del dataset
    if argumentos.sin_vecindad:
        conf.usar_vecindad = False
    if argumentos.sin_posicion:
        conf.usar_posicion = False
    if argumentos.modo_puntaje:
        conf.modo_puntaje = argumentos.modo_puntaje
    if argumentos.regimen:
        conf.regimen = argumentos.regimen
    if argumentos.disposicion:
        conf.disposicion = argumentos.disposicion
    if argumentos.beta_neg is not None:
        conf.beta_neg = argumentos.beta_neg
    if argumentos.razon_purga is not None:
        conf.razon_purga = argumentos.razon_purga
    if conf.regimen == "oneclass":
        # Sin anomalas etiquetadas los tres componentes carecen de entrada; se
        # anulan de forma explicita para que el log refleje la configuracion real.
        conf.beta_neg, conf.razon_purga = 0.0, 0.0
        conf.modo_umbral = "percentile"
    fijar_semilla(conf.semilla)
    registro = configurar_registro(conf.dir_salida)

    categorias = (categorias_de(conf) if argumentos.categorias == "all"
            else tuple(c.strip() for c in argumentos.categorias.split(",")))

    comp = "+".join([c for c, activo in (("N", conf.usar_vecindad),
                                     ("P", conf.usar_posicion)) if activo]) or "baseline"
    registro.info(f"=== PNI [{comp}] | dataset={conf.conjunto_datos} | device={conf.dispositivo} "
                f"| backbone={conf.backbone} "
                f"| capas={conf.capas} fusion={conf.capa_fusion} "
                f"| res={conf.tam_recorte} "
                f"| C_emb={conf.razon_coreset:.0%} C_dist={conf.tam_banco_dist} "
                f"| p_vecindad={conf.vecindad_dist} "
                f"| umbral={conf.metodo_honesto}"
                f"{f'(p{conf.percentil_honesto:.0f})' if conf.metodo_honesto == 'percentile' else ''}"
                f"{f'(k{conf.k_honesto:g})' if conf.metodo_honesto == 'sigma' else ''} "
                f"| seed={conf.semilla} ===")

    cerrojo = adquirir_cerrojo(conf, comp, registro)
    if cerrojo is None:
        return

    # extractor: se construye UNA vez (backbone congelado, comun a todas las categorias)
    extractor = ExtractorJerarquico(
        conf.backbone, conf.capas, conf.capa_fusion, conf.vecindad, conf.dispositivo)

    filas = []
    t_total = time.time()
    campos = ["category", "auroc_image", "auroc_pixel", "bestF1_f1",
              "bestF1_precision", "bestF1_recall", "honest_f1",
              "honest_precision", "honest_recall", "n_val", "n_coreset",
              "n_neg", "n_purged"]
    parcial = os.path.join(conf.dir_salida, f"parcial_{conf.conjunto_datos}_{comp}.csv")

    hechas = {}
    if argumentos.reanudar and os.path.exists(parcial):
        with open(parcial, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["category"] == "PROMEDIO":
                    continue
                fila = dict(r)
                for k, v in r.items():
                    if k == "category":
                        continue
                    elif k in ("n_val", "n_coreset", "n_neg", "n_purged"):
                        fila[k] = int(v) if v not in ("", None) else v
                    else:
                        fila[k] = float(v)
                hechas[r["category"]] = fila
        if hechas:
            registro.info(f"--reanudar: {len(hechas)} categorias ya evaluadas en "
                        f"{os.path.basename(parcial)} -> {sorted(hechas)}")

    for categoria in categorias:
        if categoria in hechas:
            r = hechas[categoria]
            filas.append(r)
            registro.info(f"[{categoria}] REUTILIZADA del parcial: "
                        f"AUROC img={r['auroc_image']:.4f} "
                        f"pix={r['auroc_pixel']:.4f}")
            continue
        try:
            filas.append(ejecutar_categoria(conf, categoria, extractor, registro,
                                     con_mapas_calor=not argumentos.sin_mapas_calor))
            # volcado incremental por si el proceso muere a mitad de run:
            # lo ya evaluado queda en disco y el run se puede reanudar.
            nuevo = not os.path.exists(parcial)
            with open(parcial, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=campos)
                if nuevo:
                    w.writeheader()
                w.writerow({k: (round(v, 5) if isinstance(v, float) else v)
                            for k, v in filas[-1].items()})
        except FileNotFoundError as e:
            registro.warning(f"[{categoria}] SALTADA: {e}")

    if not filas:
        registro.error("No se evaluo ninguna categoria. Revisa la ruta del dataset.")
        return

    filas.append(fila_promedio(filas))

    # CSV
    ruta_csv = os.path.join(conf.dir_salida, f"resultados_{conf.conjunto_datos}.csv")
    campos = ["category", "auroc_image", "auroc_pixel", "bestF1_f1",
              "bestF1_precision", "bestF1_recall", "honest_f1",
              "honest_precision", "honest_recall", "n_val", "n_coreset",
              "n_neg", "n_purged"]
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for r in filas:
            w.writerow({k: (round(r[k], 5) if isinstance(r[k], float) else r[k])
                        for k in campos})

    prom = filas[-1]
    # 5 decimales: con la media rozando 0.99, el 4o decimal no basta para saber
    # de que lado del objetivo cae.
    registro.info(f"=== PROMEDIO AUROC img={prom['auroc_image']:.5f} "
                f"pix={prom['auroc_pixel']:.5f} | tiempo total {time.time()-t_total:.1f}s ===")
    # Contraste con la literatura. La referencia se RECALCULA sobre las mismas
    # categorias que se han evaluado (modulo modelo_semisupervisado.referencias): comparar
    # un promedio de doce categorias contra el promedio publicado de quince no
    # seria valido, porque bastaria con que las excluidas fueran las dificiles
    # para que la diferencia se debiera al reparto y no al metodo.
    comparar_con_referencia(conf, filas, comp, registro)
    registro.info(f"CSV guardado en: {ruta_csv}")
    # Informe de generalizacion (R6). Solo tiene sentido sobre MVTec con la
    # particion de R4, que es donde hay un tipo de defecto reservado.
    if conf.conjunto_datos == "mvtec" and conf.disposicion == "split" and len(filas) > 2:
        try:
            G.informe(conf.dir_salida, conf.conjunto_datos)
        except Exception as e:                  # nunca debe tumbar un run de 4 h
            registro.warning("No se pudo generar el informe de R6: %s", e)
    try:
        os.remove(cerrojo)
    except OSError:
        pass


if __name__ == "__main__":
    principal()

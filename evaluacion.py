'''
Modulo (e): evaluacion (5 metricas) + heatmaps de localizacion.

METRICAS
  - AUROC imagen : roc_auc_score(labels, puntajes_imagen). Libre de umbral -> sin leakage.
  - AUROC pixel  : aplana todos los mapas de pixeles y todas las mascaras y calcula
                   un unico roc_auc_score. Libre de umbral.
  - F1 / precision / recall : NECESITAN un umbral binario. Aqui esta el punto
                   delicado del data leakage, por eso reportamos DOS variantes:

    (A) F1-OPTIMO sobre la curva PR de test. Recorre todos los umbrales y toma el
        de mayor F1. Es lo que reporta casi toda la literatura -> comparable.
        OJO: usa las etiquetas de test para elegir el punto, no es un umbral
        desplegable. Se declara como "F1 maximo sobre la curva precision-recall".

    (B) UMBRAL HONESTO (desplegable, sin leakage). Se calibra SOLO con los scores
        de imagenes good de VALIDACION (apartadas en el modulo a, nunca en el banco).
        Por defecto, PERCENTIL:
            tau = percentil_p(scores good de validacion)      # p=90 por defecto
        El percentil se fija A PRIORI como la tasa de falsa alarma tolerada
        (p=90 -> 10% de falsas alarmas admitidas), NO se ajusta mirando el test.
        Se descarto `media + k*desv` como opcion por defecto porque asume
        normalidad y los scores medidos son asimetricos (ver umbral_honesto);
        con k=3 el umbral quedaba tan alto que el recall caia a 0.26 en screw.
        La variante sigma sigue disponible (method="sigma") para la ablation.
        Luego se aplica ese tau fijo al test. Respeta la restriccion de no calibrar
        con anomalas.

  Recomendacion para la tesis: reporta ambas. (A) para comparar; (B) para demostrar
  que entiendes el leakage y que el sistema es desplegable.

HEATMAPS
  Superpone el mapa de pixeles sobre la imagen original (denormalizada) junto a la
  mascara ground-truth. Se generan para los casos mas representativos (anomalias
  mejor puntuadas) del capitulo de resultados.
'''
import os
from glob import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # backend sin ventana (para guardar PNG)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve

from modelo_semisupervisado.conjuntos_datos import IMAGENET_MEDIA, IMAGENET_DESV


# %% metricas

def auroc_imagen(puntajes_imagen, etiquetas):
    return float(roc_auc_score(etiquetas, puntajes_imagen))


def auroc_pixel(mapas_pixeles, mascaras):
    mapas_planos = np.concatenate([m.ravel() for m in mapas_pixeles])
    mascaras_planas = np.concatenate([m.ravel() for m in mascaras]).astype(int)
    return float(roc_auc_score(mascaras_planas, mapas_planos))


# %% Umbrales de decision

def umbral_mejor_f1(puntajes_imagen, etiquetas):
    '''(A) Mejor umbral por F1 sobre la curva PR de test.'''
    puntajes = np.asarray(puntajes_imagen, dtype=float)
    etiquetas = np.asarray(etiquetas, dtype=int)
    prec, rec, umbral = precision_recall_curve(etiquetas, puntajes)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    mejor = int(np.argmax(f1[:-1]))          # el ultimo punto no tiene umbral
    return float(umbral[mejor])


def umbral_honesto(puntajes_validacion, metodo="percentile", k=3.0, percentil=90.0):
    '''
    (B) Umbral calibrado SOLO con las good de validacion. Nunca ve anomalas.

    method="percentile" (POR DEFECTO):
        tau = percentil_p de los scores good de validacion.
        El parametro se elige A PRIORI como la TASA DE FALSA ALARMA tolerada:
        p=90 -> se acepta un 10% de falsas alarmas sobre piezas buenas.
        Es una decision de ingenieria declarada de antemano, NO ajustada sobre
        test (eso seria leakage).
        Ventaja sobre sigma: no presupone que los scores sean gaussianos.

    method="sigma" (variante clasica, se mantiene para la ablation):
        tau = media + k*desv.
        OJO: asume NORMALIDAD. Medido en este dataset, los scores good tienen
        asimetria notable (pill +1.54, grid +0.80), asi que mu+k*sigma empuja el
        umbral demasiado arriba y sacrifica recall (con k=3: recall 0.26 en screw,
        0.35 en pill, con precision 1.00 -> casi no marca nada).

    method="max":
        tau = max(val). Equivale a "cero falsas alarmas en validacion"; muy
        sensible al tamano de la muestra.

    LIMITACION A DECLARAR: con pocas good de validacion (p.ej. toothbrush n=6)
    cualquier estimador de cola es fragil. Ver `n_val` en el CSV.
    '''
    val = np.asarray(puntajes_validacion, dtype=float)
    if metodo == "percentile":
        return float(np.percentile(val, percentil))
    if metodo == "sigma":
        return float(val.mean() + k * val.std())
    if metodo == "max":
        return float(val.max())
    raise ValueError(f"metodo de umbral desconocido: {metodo!r}")


def metricas_clasificacion(puntajes_imagen, etiquetas, umbral):
    '''precision / recall / F1 dado un umbral (imagen anomala si score >= tau).'''
    puntajes = np.asarray(puntajes_imagen, dtype=float)
    etiquetas = np.asarray(etiquetas, dtype=int)
    predicciones = (puntajes >= umbral).astype(int)
    tp = int(((predicciones == 1) & (etiquetas == 1)).sum())
    fp = int(((predicciones == 1) & (etiquetas == 0)).sum())
    fn = int(((predicciones == 0) & (etiquetas == 1)).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {"threshold": float(umbral), "precision": precision,
            "recall": recall, "f1": f1}


def evaluar(puntajes_imagen, etiquetas, mapas_pixeles, mascaras, puntajes_validacion,
             metodo="percentile", k=3.0, percentil=90.0,
             umbral_forzado=None):
    '''Devuelve un dict con TODAS las metricas de una categoria.

    `umbral_forzado` sustituye al umbral honesto por uno calibrado fuera,
    con las anomalas ETIQUETADAS DE ENTRENAMIENTO (regimen semisupervisado).
    Sigue sin haber fuga: esas imagenes no estan en el conjunto de prueba, cosa
    que el resultado R4 verifica de forma explicita. El umbral best-F1 se
    conserva en paralelo porque es el que reporta la literatura y hace falta
    para comparar, aunque se calcule sobre el test y por tanto no sea honesto.
    '''
    res = {
        "auroc_image": auroc_imagen(puntajes_imagen, etiquetas),
        "auroc_pixel": auroc_pixel(mapas_pixeles, mascaras),
        "n_val": int(len(puntajes_validacion)),   # transparencia: fiabilidad del umbral
    }
    tau_a = umbral_mejor_f1(puntajes_imagen, etiquetas)
    res["metrics_bestF1"] = metricas_clasificacion(puntajes_imagen, etiquetas, tau_a)
    if umbral_forzado is not None:
        tau_b = float(umbral_forzado)
    else:
        tau_b = umbral_honesto(puntajes_validacion, metodo=metodo, k=k, percentil=percentil)
    res["metrics_honest"] = metricas_clasificacion(puntajes_imagen, etiquetas, tau_b)
    res["threshold_honest"] = tau_b
    return res


# %% heatmaps

def _desnormalizar_imagen(tensor_imagen):
    '''Deshace la normalizacion ImageNet para visualizar. (1,3,H,W)->(H,W,3) en [0,1].'''
    media = np.array(IMAGENET_MEDIA).reshape(3, 1, 1)
    desv = np.array(IMAGENET_DESV).reshape(3, 1, 1)
    imagen = tensor_imagen.squeeze(0).cpu().numpy() * desv + media
    return np.clip(imagen.transpose(1, 2, 0), 0, 1)


def guardar_mapa_calor(tensor_imagen, mapa_pixeles, mascara, puntaje, ruta):
    '''Panel de 3: imagen | mapa de anomalia superpuesto | ground-truth.'''
    pm = mapa_pixeles - mapa_pixeles.min()
    pm = pm / (pm.max() + 1e-12)             # normaliza SOLO para visualizar
    fig, eje = plt.subplots(1, 3, figsize=(12, 4))
    eje[0].imshow(_desnormalizar_imagen(tensor_imagen)); eje[0].set_title("Imagen")
    eje[1].imshow(_desnormalizar_imagen(tensor_imagen))
    eje[1].imshow(pm, cmap="jet", alpha=0.5)
    eje[1].set_title(f"Mapa de anomalia (score={puntaje:.2f})")
    eje[2].imshow(mascara, cmap="gray"); eje[2].set_title("Ground truth")
    for a in eje:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(ruta, bbox_inches="tight", dpi=120)
    plt.close(fig)


def generar_mapas_calor(evaluador, extractor, cargador_prueba, puntajes_imagen, etiquetas,
                      dir_salida, categoria, n=3):
    '''Genera heatmaps de las n anomalias mejor puntuadas de la categoria.

    Antes de generar, BORRA los heatmaps previos de esta categoria. El nombre
    del fichero lleva el indice de la imagen elegida, y ese indice cambia con la
    configuracion (otra fusion -> otros scores -> otras "peores" anomalias). Sin
    esta limpieza se acumulan heatmaps de configuraciones distintas en la misma
    carpeta, con el riesgo de publicar en la tesis una figura que NO corresponde
    a la configuracion declarada.
    '''
    os.makedirs(dir_salida, exist_ok=True)
    for viejo in glob(os.path.join(dir_salida, f"{categoria}_heatmap_*.png")):
        os.remove(viejo)
    puntajes = np.asarray(puntajes_imagen)
    etiquetas = np.asarray(etiquetas)
    indices_anomalas = np.where(etiquetas == 1)[0]
    elegido = set(indices_anomalas[np.argsort(puntajes[indices_anomalas])[::-1][:n]].tolist())

    for i, (imagen, etiqueta, mascara) in enumerate(cargador_prueba):
        if i not in elegido:
            continue
        s, mapa_p = evaluador.puntuar(extractor, imagen)
        ruta = os.path.join(dir_salida, f"{categoria}_heatmap_{i}.png")
        guardar_mapa_calor(imagen, mapa_p, mascara.squeeze().numpy(), s, ruta)


if __name__ == "__main__":
    # Prueba de humo SIN dataset: metricas con datos sinteticos.
    #   python -m modelo_semisupervisado.evaluacion
    generador = np.random.default_rng(0)
    n_conformes, n_anomalas = 60, 40
    puntajes = np.concatenate([generador.normal(1.0, 0.3, n_conformes),
                             generador.normal(2.5, 0.5, n_anomalas)])
    etiquetas = np.array([0] * n_conformes + [1] * n_anomalas)
    puntajes_validacion = generador.normal(1.0, 0.3, 20)                 # good de validacion
    H = W = 32
    mapas_pixeles = [generador.random((H, W)) for _ in range(n_conformes + n_anomalas)]
    mascaras = [np.zeros((H, W)) for _ in range(n_conformes)] +\
            [(generador.random((H, W)) > 0.7).astype(float) for _ in range(n_anomalas)]
    # correlacionamos un poco mapa y mascara para que la AUROC sea > 0.5
    for j in range(n_conformes, n_conformes + n_anomalas):
        mapas_pixeles[j] = mapas_pixeles[j] + mascaras[j]

    res = evaluar(puntajes, etiquetas, mapas_pixeles, mascaras, puntajes_validacion, k=3.0)
    print("AUROC imagen:", round(res["auroc_image"], 4))
    print("AUROC pixel :", round(res["auroc_pixel"], 4))
    print("best-F1  :", {k: round(v, 3) for k, v in res["metrics_bestF1"].items()})
    print("honesto  :", {k: round(v, 3) for k, v in res["metrics_honest"].items()})

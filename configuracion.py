'''
Configuracion centralizada del pipeline PNI (modulo f).

PNI (Bae et al., ICCV 2023) extiende el esquema de banco de memoria anadiendo
DOS fuentes de informacion que el banco puro ignora:

  N (neighborhood): que features son plausibles dado lo que hay ALREDEDOR.
  P (position):     que features son plausibles dado DONDE estamos en la imagen.

El score deja de ser "distancia al vecino mas cercano del banco" y pasa a ser
"distancia al vecino mas cercano ENTRE LOS CANDIDATOS PLAUSIBLES en esa
posicion y vecindad". Un parche que se parece mucho a un vector del banco que
nunca aparece ahi deja de considerarse normal.

Ablation del paper (Tabla 3, MVTec AD, I-AUROC / P-AUROC):
    sin N ni P      98.92 / 98.18   <- equivale a nuestro baseline anterior
    + N             99.44 / 98.62
    + N + P         99.46 / 98.80   <- objetivo de esta implementacion
    + N + P + R     99.56 / 98.98   <- R = red de refinamiento (no implementada)

La red de refinamiento R aporta solo +0.10 / +0.18 y exige un DenseNet161
entrenado de forma supervisada sobre cuatro tipos de anomalias sinteticas
(CutPaste, CutPaste-scar, DRAEM y dibujo manual). Se documenta como fase
opcional: su coste no compensa la ganancia dentro del alcance del proyecto.
'''
from dataclasses import dataclass
import os
import random
import numpy as np
import torch


# Categorias oficiales de MVTec AD (15). Se evaluan todas y se promedia.
# La delimitacion del alcance excluye productos alimenticios, medicos y
# farmaceuticos, asi que quedan fuera `capsule` y `pill` (farmaceuticos) y
# `hazelnut` (alimenticio). Las quince originales se conservan abajo porque las
# tablas de referencia de la literatura promedian sobre todas ellas y hay que
# poder recalcular esos promedios sobre el mismo subconjunto.
MVTEC_CATEGORIAS = (
    "bottle", "cable", "carpet", "grid", "leather", "metal_nut",
    "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)
MVTEC_CATEGORIAS_COMPLETO = MVTEC_CATEGORIAS + ("capsule", "hazelnut", "pill")
MVTEC_EXCLUIDAS_ALCANCE = ("capsule", "hazelnut", "pill")

# Categorias oficiales de VisA (12). Los tres dominios son los de la Tabla 1 de
# Zou et al. (ECCV 2022) y NO son decorativos: PNI reporta su Tabla 6 desglosada
# por ellos, asi que agruparlas mal invalidaria la comparacion.
#   estructura compleja  -> placas de circuito impreso
#   multiples instancias -> varias piezas por imagen, en posiciones y poses
#                           distintas (capsules y macaroni2 son las que mas
#                           varian de ubicacion, y las mas dificiles)
#   instancia unica      -> una pieza por imagen, aproximadamente alineada
# Tras la exclusion por alcance solo sobreviven `candle` (multiples instancias)
# y las cuatro placas de circuito impreso: el dominio de instancia unica
# desaparece entero, porque sus cuatro categorias son productos alimenticios.
VISA_CATEGORIAS = (
    "candle",                                           # multiples instancias
    "pcb1", "pcb2", "pcb3", "pcb4",                     # estructura compleja
)
VISA_CATEGORIAS_COMPLETO = (
    "candle", "capsules", "macaroni1", "macaroni2",
    "cashew", "chewinggum", "fryum", "pipe_fryum",
    "pcb1", "pcb2", "pcb3", "pcb4",
)
VISA_EXCLUIDAS_ALCANCE = ("capsules", "cashew", "chewinggum", "fryum",
                          "macaroni1", "macaroni2", "pipe_fryum")

# Dominios de la Tabla 1 de Zou et al., restringidos a lo que queda en alcance.
VISA_DOMINIOS = {
    "multiples instancias": ("candle",),
    "estructura compleja":  ("pcb1", "pcb2", "pcb3", "pcb4"),
}


# %% Hiperparametros del pipeline

@dataclass
class Configuracion:
    # Datos (modulo a)
    conjunto_datos: str = "mvtec"         # "mvtec" | "visa"
    # Una raiz por dataset. raiz_datos se RESUELVE a partir de `conjunto_datos` en
    # resolver_rutas(), de modo que cambiar --conjunto-datos baste para cambiar de
    # corpus sin editar rutas a mano (fuente clasica de runs cruzados).
    raiz_mvtec: str = r"C:\Users\USER\Desktop\tesis\modelo_semisupervisado\data\datasets\mvtec"
    raiz_visa: str = ""            # "" -> autodeteccion (datasets/visa o datasets/archive)
    raiz_datos: str = r"C:\Users\USER\Desktop\tesis\modelo_semisupervisado\data\datasets\mvtec"
    # PNI opera a 512 -> 480 (no a 256 -> 224). A 480 px la rejilla de layer2
    # es 60x60, que es la resolucion a la que PNI fusiona las escalas.
    tam_imagen: int = 512          # Resize antes del crop
    tam_recorte: int = 480           # CenterCrop -> entrada al backbone
    razon_validacion: float = 0.10        # good apartadas para umbral honesto (modulo e)
    tam_lote: int = 4            # bajado de 8: a 480 px cada lote pesa 4x mas
    n_trabajadores: int = 0           # 0 en Windows (evita fallos de spawn)

    # Extractor (modulo b)
    backbone: str = "wide_resnet50_2"
    capas: tuple = ("layer1", "layer2", "layer3")  # >=3 niveles -> OE2
    # PNI fusiona a la resolucion de layer2 (h* = max(h2,h3)). A 480 px eso son
    # 60x60 = 3600 parches por imagen. Fusionar a layer1 (120x120) cuadruplicaria
    # el banco hasta ~2.9M vectores por categoria: inviable en 16 GB de RAM.
    capa_fusion: str = "layer2"
    vecindad: int = 3          # avgpool p x p para features locales

    # Banco de memoria (modulo c)
    # C_emb: coreset de embedding, el banco clasico contra el que se mide la
    # distancia. El paper fija la razon de submuestreo en 0.01 (seccion 4.1).
    # A 60x60 con ~220 imagenes hay ~790k parches; 1% deja ~7.9k.
    razon_coreset: float = 0.01
    # C_dist: coreset de distribucion, SUBMUESTREO DE C_emb. Es el alfabeto
    # sobre el que el MLP y el histograma predicen probabilidades.
    # El paper NO usa una fraccion sino un TAMANO FIJO: "the size of the
    # distribution coreset |C_dist| is set to 2,048" (seccion 4.1). Que sea
    # fijo importa: |C_dist| es el numero de clases del MLP, y dejarlo
    # proporcional a |C_emb| haria que la dificultad del problema de
    # clasificacion cambiase de una categoria a otra.
    tam_banco_dist: int = 2048
    dim_proyeccion: int = 128      # Johnson-Lindenstrauss para acelerar

    # Topes de memoria (no estan en el paper; los impone el hardware)
    # VisA tiene hasta 900 imagenes de entrenamiento por categoria. A 60x60 eso
    # son 2,9 M de parches x 1792 dim x 4 bytes = 20,9 GB, imposible en 16 GB.
    # Dos medidas, ambas declaradas A PRIORI (no ajustadas mirando resultados):
    #   1) los parches se guardan en float16 (la seleccion del coreset solo
    #      compara distancias; el banco final se reconvierte a float32).
    #   2) el conjunto CANDIDATO del greedy se limita por muestreo aleatorio
    #      uniforme con semilla fija. El tamano del coreset m se sigue
    #      calculando sobre el numero REAL de parches (razon_coreset * N_total),
    #      asi que el banco no encoge por el tope: solo se elige entre menos
    #      candidatos.
    max_parches_banco: int = 700_000
    max_coreset: int = 30_000      # tope duro de |C_emb| (coste de inferencia)

    # Regimen de supervision (OE3)
    # "oneclass" -> solo train/good.        "semi" -> train/good + train/anomaly.
    # Con regime="semi" y los tres pesos de abajo a cero, el sistema reproduce
    # EXACTAMENTE el comportamiento one-class. Eso no es cosmetico: es lo que
    # permite atribuir cualquier diferencia a las etiquetas y no al reparto de
    # los datos, ya que ambos regimenes comparten las mismas conformes y el
    # mismo conjunto de prueba (ver R4).
    regimen: str = "semi"
    # "split" -> lee data/datasets/tesis_split (particion de R4, por defecto).
    # "original" -> lee los corpus sin repartir, para reproducir los
    # resultados anteriores al giro semisupervisado.
    disposicion: str = "split"
    raiz_particion: str = r"C:\Users\USER\Desktop\tesis\modelo_semisupervisado\data\datasets\tesis_split"

    # (1) banco negativo C_neg
    # Una imagen anomala es MAYORITARIAMENTE normal: sus parches de fondo son
    # indistinguibles de los del banco positivo. Solo se toman como negativos
    # los parches cuya celda de la rejilla esta cubierta por la mascara al menos
    # en esta fraccion. Con un umbral bajo el banco negativo se llena de parches
    # sanos y deja de discriminar; con uno muy alto se queda sin muestras en los
    # defectos pequenos, que son justo los que interesan.
    cobertura_mascara_neg: float = 0.30
    max_banco_neg: int = 4096       # tope de |C_neg| tras el coreset

    # (2) depuracion supervisada del banco positivo (idea de SoftPatch)
    # SoftPatch pondera los parches del banco para descartar los que no
    # representan normalidad, estimando el ruido sin supervision (LOF). Aqui la
    # senal es directa: se descarta el parche de C_emb que esta mas cerca de
    # C_neg que de la nube normal. Son parches que TAPAN defectos conocidos:
    # mientras esten en el banco, un defecto de ese tipo siempre encuentra
    # vecino cercano y su distancia no sube.
    # Valor 0.0 -> no se depura nada (equivale al banco de PNI sin tocar).
    razon_purga: float = 0.02      # fraccion maxima de C_emb que puede purgarse
    margen_purga: float = 1.0      # se purga si d(p, C_neg) < margin * d(p, C_emb\p)

    # (3) score contrastivo
    # s = d+ - beta * d-, acotado a >= 0. Un parche que se parece a lo normal Y
    # ADEMAS se parece a un defecto conocido deja de considerarse normal.
    # beta = 0.0 -> score de PNI sin modificar.
    beta_neg: float = 0.30

    # (4) umbral de decision calibrado con las dos clases
    # Con anomalas etiquetadas DE ENTRENAMIENTO el punto de operacion puede
    # fijarse sin mirar el test, cosa que el percentil de las conformes no
    # permitia. Ataca el desajuste de la tasa de falsa alarma observado antes.
    modo_umbral: str = "balanced"   # "balanced" (usa C_neg) | "percentile"

    # Distribucion condicionada (modulo nuevo)
    usar_vecindad: bool = True  # componente N de la ablation
    usar_posicion: bool = True      # componente P de la ablation

    # TAMANO DE LA VECINDAD N_p(x). Es el parametro mas delicado de todo PNI.
    # El paper (seccion 4.1): "the patch size of the neighborhood p is set to 9".
    #
    # Por que NO puede ser 3: las features ya pasan por un AvgPool2d de 3x3
    # (conf.vecindad) antes de entrar aqui. Con p=3, las 8 features vecinas
    # comparten 6 de sus 9 celdas de origen con la del centro, asi que predecir
    # el centro a partir de ellas es casi copiarlo: el MLP aprende la identidad,
    # p(c|N) sale concentrada en el simbolo que ya era el vecino mas cercano y
    # el filtro T_tau no veta nada util. Con p=9 la ventana abarca 9/60 = 15%
    # del ancho de la imagen y el centro pasa a estar realmente PREDICHO POR SU
    # CONTEXTO, que es el mecanismo del que sale la ganancia publicada.
    vecindad_dist: int = 9

    # MLP que estima p(c_dist | N_p(x)). Entrada: las (p*p - 1) features vecinas
    # concatenadas -> (81-1) * 1792 = 143.360 dimensiones; salida: |C_dist|.
    # El paper usa N_MLP=4 capas de c_MLP=2048 neuronas, 15 epocas, Adam lr=1e-3,
    # batch 2048 y decaimiento escalonado gamma=0.1 cada 5 epocas.
    # DESVIACION DECLARADA: 2048 neuronas en la primera capa son 143360*2048 =
    # 294 M de parametros; con los estados de Adam eso pide ~4,7 GB de VRAM y no
    # deja sitio para el resto del pipeline en una GPU de 8 GB. Se usan 1024
    # (147 M de parametros, ~2,4 GB) y batch 512 en vez de 2048.
    ocultas_mlp: int = 1024
    capas_mlp: int = 4            # capas secuenciales (N_MLP en el paper)
    epocas_mlp: int = 15
    tasa_mlp: float = 1e-3
    paso_tasa_mlp: int = 5           # StepLR: cada 5 epocas
    gamma_tasa_mlp: float = 0.1      # ...multiplica lr por 0.1
    lote_mlp: int = 512
    # Fraccion de posiciones por imagen usadas en cada epoca del MLP. A 60x60
    # hay 3600 posiciones por imagen y las vecindades solapan mucho, asi que
    # usarlas todas cada epoca es redundante. Cada epoca sortea posiciones
    # distintas, de modo que a lo largo del entrenamiento se cubren todas.
    submuestreo_pos_mlp: float = 0.25
    # Tope de muestras por epoca. Sin el, una categoria de VisA (900 imagenes)
    # entrenaria con 810.000 vecindades por epoca frente a las 173.000 de una
    # de MVTec: el MLP veria 4,7 veces mas datos en VisA y las dos mitades del
    # estudio dejarian de ser comparables. Fijarlo iguala el presupuesto de
    # entrenamiento entre datasets y ademas acota el tiempo por categoria.
    muestras_por_epoca_mlp: int = 250_000
    # Numero de posiciones cuyas vecindades se materializan a la vez. Con p=9 y
    # 1792 canales, cada vecindad son 143.360 floats = 574 KB: materializar las
    # 3600 posiciones de una imagen de golpe pediria 2,1 GB.
    trozo_vecindad: int = 256
    # Temperature scaling (Guo et al.): T=2 en el paper. Corrige el exceso de
    # confianza tipico de una red profunda entrenada con cross-entropy; sin el,
    # p(c|N) satura en 0/1 y el umbral tau deja de filtrar nada.
    temperatura: float = 2.0

    # Scoring (modulo d)
    # Forma de agregar los candidatos del banco:
    #   "max" -> eq. 4 del paper: distancia al candidato plausible mas cercano.
    #   "sum" -> eq. 3: suma ponderada por p(c|Omega), continua.
    # El paper declara usar "max" como aproximacion rapida de "sum", admitiendo
    # "una pequena perdida de rendimiento". Cual rinde mas en esta
    # implementacion es una pregunta empirica: se mide, no se asume.
    modo_puntaje: str = "max"
    # p(Phi | c) ~ exp(-lambda * ||Phi - c||). El paper fija lambda=1 sin optimizar.
    lambda_exp: float = 1.0
    # Umbral de plausibilidad: T_tau(x) = 1 si x > tau. El paper usa
    # tau = 1 / (2 * |C_emb|), que garantiza que al menos un candidato sobreviva.
    factor_tau: float = 0.5        # tau = factor_tau / |C_emb|
    # PNI suaviza con sigma=8 (nuestro pipeline anterior usaba 4, pero a 224 px;
    # a 480 px la escala del mapa se duplica, asi que sigma tambien).
    sigma_gaussiano: float = 8.0
    n_mapas_calor: int = 3            # heatmaps por categoria

    # Umbral honesto (e): se calibra SOLO con good de validacion
    metodo_honesto: str = "percentile"
    percentil_honesto: float = 90.0
    k_honesto: float = 3.0          # solo si metodo_honesto="sigma"

    # Reproducibilidad
    semilla: int = 0
    dispositivo: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Salidas
    dir_salida: str = r"C:\Users\USER\Desktop\tesis\modelo_semisupervisado\outputs"


# %% Resolucion de rutas y semillas

def categorias_de(conf) -> tuple:
    '''Lista de categorias del dataset activo.'''
    return VISA_CATEGORIAS if conf.conjunto_datos == "visa" else MVTEC_CATEGORIAS


def _autodetectar_visa(base) -> str:
    '''Localiza la raiz de VisA.

    Se acepta tanto datasets/visa (nombre canonico) como datasets/archive (el
    nombre con el que sale la descarga oficial sin renombrar). Se
    reconoce por la presencia de split_csv/1cls.csv, que es el fichero que
    define el split oficial de una clase; sin el no se puede evaluar VisA de
    forma comparable con la literatura.
    '''
    for nombre in ("visa", "VisA", "archive"):
        cand = os.path.join(base, nombre)
        if os.path.exists(os.path.join(cand, "split_csv", "1cls.csv")):
            return cand
    return os.path.join(base, "visa")


def resolver_rutas(conf) -> "Configuracion":
    '''Fija conf.raiz_datos segun conf.conjunto_datos. Idempotente.'''
    base = os.path.dirname(conf.raiz_mvtec)          # .../tesis/modelo_semisupervisado/data/datasets
    if conf.conjunto_datos == "visa":
        conf.raiz_datos = conf.raiz_visa or _autodetectar_visa(base)
    else:
        conf.raiz_datos = conf.raiz_mvtec
    return conf


def fijar_semilla(semilla: int = 0):
    '''Fija TODAS las fuentes de aleatoriedad + fuerza determinismo en cuDNN.'''
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def obtener_configuracion() -> Configuracion:
    conf = resolver_rutas(Configuracion())
    os.makedirs(conf.dir_salida, exist_ok=True)
    return conf

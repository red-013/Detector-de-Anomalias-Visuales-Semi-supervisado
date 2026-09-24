'''
Medio de verificacion del resultado R2: evidencia de la extraccion jerarquica.

Genera dos artefactos que demuestran que el modelo extrae representaciones en
al menos tres niveles de abstraccion distintos:

  1) `niveles_dimensiones.txt`  tabla con el numero de canales y la resolucion
     de cada nivel, y con la dimension del descriptor tras la fusion.
  2) `niveles_<dataset>_<categoria>_<n>.png`  visualizacion de los mapas de
     caracteristicas de cada nivel para una muestra conforme y otra anomala.

Cada mapa se resume por la NORMA L2 a lo largo de los canales, que da la
magnitud de activacion en cada posicion. Es preferible a promediar los canales:
las activaciones de una ResNet son no negativas pero de escalas muy distintas
entre canales, y la media queda dominada por unos pocos, mientras que la norma
refleja cuanta senal hay en esa posicion con independencia de que canal la
aporte. Cada nivel se normaliza a [0, 1] por separado, ya que la magnitud
absoluta crece con la profundidad y sin normalizar solo se veria layer3.

Se ejecuta en CPU a proposito: es un artefacto de documentacion y no debe
competir por la GPU con un run de evaluacion en marcha.

Uso:
    .\\venv\\Scripts\\python.exe -m modelo_semisupervisado.visualizar_niveles
    .\\venv\\Scripts\\python.exe -m modelo_semisupervisado.visualizar_niveles --conjunto-datos visa --categoria pcb1
'''
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from modelo_semisupervisado.configuracion import obtener_configuracion
from modelo_semisupervisado.conjuntos_datos import ConjuntoParticionTesis, IMAGENET_MEDIA, IMAGENET_DESV
from modelo_semisupervisado.extractor_caracteristicas import ExtractorJerarquico


# %% Preparacion de las imagenes

def _desnormalizar(t):
    m = np.array(IMAGENET_MEDIA).reshape(3, 1, 1)
    s = np.array(IMAGENET_DESV).reshape(3, 1, 1)
    x = t.squeeze(0).cpu().numpy() * s + m
    return np.clip(x.transpose(1, 2, 0), 0, 1)


@torch.no_grad()
def mapas_por_nivel(extractor, imagen, capas):
    '''Norma L2 por posicion de cada nivel, en su resolucion NATIVA.'''
    extractor._caracteristicas.clear()
    extractor.backbone(imagen)
    salida = {}
    for n in capas:
        f = extractor._caracteristicas[n]                       # (1, C, H, W)
        salida[n] = f.squeeze(0).norm(dim=0).cpu().numpy()
    return salida


# %% Composicion de la figura

def figura(imagen, mapas, capas, mascara, titulo, destino):
    n = len(capas) + (2 if mascara is not None else 1)
    fig, ejes = plt.subplots(1, n, figsize=(3.1 * n, 3.4))
    ejes[0].imshow(_desnormalizar(imagen))
    ejes[0].set_title("entrada\n%dx%d px" % (imagen.shape[2], imagen.shape[3]), fontsize=9)
    ejes[0].axis("off")
    for i, c in enumerate(capas, start=1):
        m = mapas[c]
        m = (m - m.min()) / (m.max() - m.min() + 1e-12)
        ejes[i].imshow(m, cmap="inferno")
        ejes[i].set_title("%s\n%dx%d" % (c, m.shape[0], m.shape[1]), fontsize=9)
        ejes[i].axis("off")
    if mascara is not None:
        ejes[-1].imshow(mascara.squeeze().cpu().numpy(), cmap="gray")
        ejes[-1].set_title("mascara\n(verdad de campo)", fontsize=9)
        ejes[-1].axis("off")
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(destino, dpi=130, bbox_inches="tight")
    plt.close(fig)


# %% Punto de entrada

def principal():
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--conjunto-datos", default="mvtec", choices=("mvtec", "visa"))
    analizador.add_argument("--categoria", default="bottle")
    analizador.add_argument("--n", type=int, default=2, help="muestras de cada clase")
    argumentos = analizador.parse_args()

    conf = obtener_configuracion()
    conf.conjunto_datos = argumentos.conjunto_datos
    conf.dispositivo = "cpu"
    ruta_destino = os.path.join(conf.dir_salida, "niveles")
    os.makedirs(ruta_destino, exist_ok=True)

    extractor = ExtractorJerarquico(conf.backbone, conf.capas, conf.capa_fusion,
                                      conf.vecindad, "cpu")

    # 1) tabla de dimensiones
    x = torch.zeros(1, 3, conf.tam_recorte, conf.tam_recorte)
    extractor._caracteristicas.clear()
    extractor.backbone(x)
    fusionado = extractor(x)
    congelados = sum(1 for p in extractor.backbone.parameters() if p.requires_grad)
    lineas = [
        "Extraccion jerarquica de caracteristicas (resultado R2)",
        "=" * 62,
        "Backbone      : %s (preentrenado en ImageNet, congelado)" % conf.backbone,
        "Parametros    : %.2f M, de los cuales %d requieren gradiente"
        % (sum(p.numel() for p in extractor.backbone.parameters()) / 1e6, congelados),
        "Entrada       : 3 x %d x %d px" % (conf.tam_recorte, conf.tam_recorte),
        "Agregacion    : AvgPool2d %dx%d, stride 1 (no altera la resolucion)"
        % (conf.vecindad, conf.vecindad),
        "",
        "%-10s %10s %14s %14s" % ("nivel", "canales", "rejilla", "posiciones"),
        "-" * 62,
    ]
    for n in conf.capas:
        f = extractor._caracteristicas[n]
        lineas.append("%-10s %10d %14s %14d"
                      % (n, f.shape[1], "%dx%d" % (f.shape[2], f.shape[3]),
                         f.shape[2] * f.shape[3]))
    lineas += [
        "-" * 62,
        "fusion en %-4s %10d %14s %14d"
        % (conf.capa_fusion, fusionado.shape[1],
           "%dx%d" % (fusionado.shape[2], fusionado.shape[3]),
           fusionado.shape[2] * fusionado.shape[3]),
        "",
        "El descriptor de cada posicion concatena los %d niveles: %s = %d dim."
        % (len(conf.capas),
           " + ".join(str(extractor._caracteristicas[n].shape[1]) for n in conf.capas),
           fusionado.shape[1]),
        "Los niveles se alinean por interpolacion bilineal a la resolucion de",
        "%s antes de concatenarse." % conf.capa_fusion,
    ]
    ruta_texto = os.path.join(ruta_destino, "niveles_dimensiones.txt")
    with open(ruta_texto, "w", encoding="utf-8") as fichero:
        fichero.write("\n".join(lineas) + "\n")
    print("\n".join(lineas))
    print("\n[OK] tabla en %s" % ruta_texto)

    # 2) visualizaciones
    for particion, clase in (("train_good", "conforme"), ("test", "anomala")):
        conj = ConjuntoParticionTesis(conf.raiz_particion, conf.conjunto_datos, argumentos.categoria, particion,
                               conf.tam_imagen, conf.tam_recorte)
        # En test se buscan anomalas; en train_good todas son conformes.
        indice = [i for i, (_, etq, _) in enumerate(conj.muestras)
               if (etq == 1 if clase == "anomala" else etq == 0)][:argumentos.n]
        for k, i in enumerate(indice):
            imagen, etq, masc = conj[i]
            imagen = imagen.unsqueeze(0)
            mapas = mapas_por_nivel(extractor, imagen, conf.capas)
            nombre = "niveles_%s_%s_%s_%d.png" % (conf.conjunto_datos, argumentos.categoria, clase, k)
            figura(imagen, mapas, conf.capas,
                   masc if clase == "anomala" else None,
                   "%s / %s - muestra %s" % (conf.conjunto_datos, argumentos.categoria, clase),
                   os.path.join(ruta_destino, nombre))
            print("[OK] %s" % nombre)


if __name__ == "__main__":
    principal()

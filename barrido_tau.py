'''Barrido de tau reutilizando banco y distribucion (una sola construccion).

PENDIENTE 2026-08-24: es la prueba que quedo a medias. Responde si la ausencia
de ganancia de N+P se debe a que tau (0.5/|C_emb|) es demasiado permisivo, o si
el problema es estructural.

Uso:  .\venv\Scripts\python.exe -m modelo_semisupervisado.barrido_tau
'''
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from modelo_semisupervisado.configuracion import obtener_configuracion, fijar_semilla
from modelo_semisupervisado.conjuntos_datos import construir_cargadores
from modelo_semisupervisado.extractor_caracteristicas import ExtractorJerarquico
from modelo_semisupervisado.banco_memoria import BancoDeMemoria
from modelo_semisupervisado.distribucion import DistribucionCondicional
from modelo_semisupervisado.puntuacion import Evaluador

categoria = "transistor"
conf = obtener_configuracion(); fijar_semilla(conf.semilla)
extractor = ExtractorJerarquico(conf.backbone, conf.capas, conf.capa_fusion,
                                  conf.vecindad, conf.dispositivo)
ent, val, pru = construir_cargadores(conf, categoria)
bm = BancoDeMemoria(conf); bm.ajustar(extractor, ent)
dist = DistribucionCondicional(conf, bm).ajustar(extractor, ent)

# cachea features y probabilidades de test UNA vez
cache = []
for imagen, etq, masc in pru:
    mapa_caract = extractor(imagen)
    _, C, H, W = mapa_caract.shape
    parches = mapa_caract.permute(0, 2, 3, 1).reshape(-1, C).to(conf.dispositivo)
    D = torch.cdist(parches, bm.banco.to(conf.dispositivo))
    pe = dist.predecir(mapa_caract)[:, bm.emb2dist.to(conf.dispositivo)]
    cache.append((D, pe, int(etq.item())))
print(f"cacheadas {len(cache)} imagenes de test")

M = bm.banco.shape[0]
print(f"\n{'tau_factor':>11} {'% vivos':>9} {'AUROC img':>10}")
base = None
for factor in [0.0, 0.5, 5, 20, 50, 100, 200, 500]:
    tau = factor / M
    puntajes, etiquetas, vivos = [], [], []
    for D, pe, etq in cache:
        d = D.masked_fill(pe <= tau, float("inf")) if factor > 0 else D
        dm = d.min(dim=1).values
        malos = ~torch.isfinite(dm)
        if malos.any():
            dm[malos] = D[malos].min(dim=1).values
        puntajes.append(dm.max().item()); etiquetas.append(etq)
        vivos.append((pe > tau).float().mean().item() if factor > 0 else 1.0)
    au = roc_auc_score(etiquetas, puntajes)
    if base is None: base = au
    print(f"{factor:>11.1f} {100*np.mean(vivos):>8.1f}% {au:>10.4f}"
          f"{'   <- baseline' if factor == 0 else f'   ({au-base:+.4f})'}")

'''
Sweep de POST-PROCESO: sigma (suavizado del mapa de pixeles) y b (vecinos de la
reponderacion del score de imagen).

POR QUE UN SCRIPT APARTE Y NO LA ABLATION NORMAL
  sigma y b actuan DESPUES de la busqueda de vecinos: no cambian el banco ni
  las distancias parche-banco. Rehacer el pipeline completo por cada valor
  (como hace ablaciones.py) pagaria ~12 min por variante para medir un filtro
  que tarda milisegundos. Aqui el pipeline corre UNA vez por categoria, se
  cachean (i) el mapa interpolado SIN suavizar y (ii) las cantidades del
  reweight, y se re-evalua cada valor sobre esa cache.

QUE RESPONDE (predicciones falsables, ver bitacora 8bis):
  - sigma in {0,1,2,3,4,6,8}: el 4.0 actual se eligio sin justificacion
    empirica y actua directo sobre la AUROC de pixel. Si wood/tile (defectos
    grandes de borde difuso) prefieren sigma grande y el resto no, la
    hipotesis "bordes difusos" gana apoyo.
  - b in {2,3,5,9} + variante SIN reponderar (score = s*): b=3 se heredo sin
    ablation; implementaciones de referencia usan valores mayores (p.ej. 9).
    Solo afecta a la AUROC de imagen.

VALIDACION INTERNA: la columna sigma=4 y la columna b=3 deben REPRODUCIR
exactamente resultados.csv (misma config, misma semilla). Si no, hay un bug.

USO
  python -m modelo_semisupervisado.barrido_posproceso
  python -m modelo_semisupervisado.barrido_posproceso --categorias wood,tile
  # con overrides de config (para medir b/sigma sobre OTRO banco):
  python -m modelo_semisupervisado.barrido_posproceso --coreset 0.10 --fusion layer2 --etiqueta c10_f28
'''
import argparse
import csv
import os
import time
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon

from modelo_semisupervisado.configuracion import obtener_configuracion, fijar_semilla, MVTEC_CATEGORIAS
from modelo_semisupervisado.conjuntos_datos import construir_cargadores
from modelo_semisupervisado.extractor_caracteristicas import ExtractorJerarquico
from modelo_semisupervisado.banco_memoria import BancoDeMemoria
from modelo_semisupervisado.evaluacion import auroc_imagen, auroc_pixel
from modelo_semisupervisado.principal import configurar_registro

SIGMAS = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
BS = [2, 3, 5, 9]           # b de la reponderacion
B_BASE = 3                  # valor actual (config.n_reweight)
SIGMA_BASE = 4.0            # valor actual (config.gaussian_sigma)


# %% Barrido de post-proceso

@torch.no_grad()
def puntuar_imagen_todos_b(parches, d_parche, ind_vecino, banco):
    '''Devuelve {'sin_reweight': s*, 'b=2': ..., ...} para UNA imagen.

    Identico a Evaluador._reweighted_image_score pero calculando el peso w para
    todos los b de una vez: los vecinos de m* se ordenan por distancia, y el
    conjunto N_b es simplemente los primeros b de esa lista.
    '''
    s_star, ind_q = d_parche.max(dim=0)
    q = parches[ind_q:ind_q + 1]
    m_star = banco[ind_vecino[ind_q]:ind_vecino[ind_q] + 1]

    d_bank = torch.cdist(m_star, banco).squeeze(0)
    ind_vec = torch.topk(d_bank, k=max(BS), largest=False).indices  # ascendente
    d_i = torch.cdist(q, banco[ind_vec]).squeeze(0)                  # (max_b,)

    salida = {"sin_reweight": s_star.item()}
    for b in BS:
        w = 1.0 - 1.0 / torch.exp(d_i[:b] - s_star).sum()
        salida[f"b={b}"] = (w * s_star).item()
    return salida


@torch.no_grad()
def mapa_crudo(d_parche, rejilla, tam_recorte):
    '''Mapa interpolado a resolucion de entrada, SIN suavizado gaussiano.'''
    Hr, Wr = rejilla
    baja_res = d_parche.reshape(1, 1, Hr, Wr)
    ampliado = F.interpolate(baja_res, size=(tam_recorte, tam_recorte),
                       mode="bilinear", align_corners=False)
    return ampliado.squeeze().cpu().numpy()


def barrer_categoria(conf, categoria, extractor, registro):
    t0 = time.time()
    (cargador_entrenamiento, cargador_validacion,
     cargador_prueba) = construir_cargadores(conf, categoria)

    objeto_banco = BancoDeMemoria(conf)
    st = objeto_banco.ajustar(extractor, cargador_entrenamiento)
    banco = objeto_banco.banco.to(conf.dispositivo)
    registro.info(f"[{categoria}] banco {st['n_full']} -> {st['n_coreset']} "
                f"(grid {st['grid']})")

    # Test: una pasada, cacheando lo necesario
    puntajes_por_b = {k: [] for k in ["sin_reweight"] + [f"b={b}" for b in BS]}
    etiquetas, crudos, mascaras = [], [], []
    for imagen, etiqueta, mascara in cargador_prueba:
        parches, rejilla = extractor.extraer_parches(imagen)
        parches = parches.to(conf.dispositivo)
        D = torch.cdist(parches, banco)
        d_parche, ind_vecino = D.min(dim=1)
        for k, v in puntuar_imagen_todos_b(parches, d_parche, ind_vecino, banco).items():
            puntajes_por_b[k].append(v)
        crudos.append(mapa_crudo(d_parche, rejilla, conf.tam_recorte))
        etiquetas.append(int(etiqueta.item()))
        mascaras.append(mascara.squeeze().numpy())

    # AUROC de imagen por b
    fila_b = {"category": categoria}
    for k, evaluador in puntajes_por_b.items():
        fila_b[k] = round(auroc_imagen(evaluador, etiquetas), 4)

    # AUROC de pixel por sigma (sigma=0 -> sin suavizar)
    fila_s = {"category": categoria}
    for s in SIGMAS:
        mapas = crudos if s == 0 else [gaussian_filter(r, sigma=s) for r in crudos]
        fila_s[f"sigma={s:g}"] = round(auroc_pixel(mapas, mascaras), 4)

    registro.info(f"[{categoria}] b: " +
                " ".join(f"{k}={fila_b[k]}" for k in puntajes_por_b) +
                f" | sigma: " +
                " ".join(f"{s:g}:{fila_s[f'sigma={s:g}']}" for s in SIGMAS) +
                f" | {time.time()-t0:.1f}s")
    return fila_b, fila_s


# %% Persistencia y contraste

def escribir_csv(ruta, filas, campos):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for r in filas:
            w.writerow(r)
        prom = {"category": "PROMEDIO"}
        for k in campos[1:]:
            prom[k] = round(sum(r[k] for r in filas) / len(filas), 4)
        w.writerow(prom)
    return prom


def wilcoxon_contra_base(filas, columna_base, columnas, etiqueta, registro):
    base = np.array([r[columna_base] for r in filas])
    for c in columnas:
        if c == columna_base:
            continue
        x = np.array([r[c] for r in filas])
        diferencia = x - base
        if np.allclose(diferencia, 0):
            registro.info(f"  {etiqueta} {c} vs {columna_base}: identicos")
            continue
        estadistico, p = wilcoxon(x, base)
        sig = "SIG" if p < 0.05 else "ns"
        registro.info(f"  {etiqueta} {c} vs {columna_base}: media {x.mean():+.4f} vs "
                    f"{base.mean():.4f} (delta {x.mean()-base.mean():+.4f}) | "
                    f"gana {int((diferencia > 0).sum())}/{len(diferencia)} | p={p:.4f} {sig}")


# %% Punto de entrada

def principal():
    analizador = argparse.ArgumentParser()
    analizador.add_argument("--categorias", type=str, default="all")
    analizador.add_argument("--coreset", type=float, default=None)
    analizador.add_argument("--fusion", type=str, default=None)
    analizador.add_argument("--recorte", type=int, default=None,
                        help="tam_recorte; tam_imagen se ajusta a recorte+32")
    analizador.add_argument("--etiqueta", type=str, default="",
                        help="subcarpeta de salida (para no pisar el sweep base)")
    argumentos = analizador.parse_args()

    conf = obtener_configuracion()
    if argumentos.coreset is not None:
        conf = replace(conf, razon_coreset=argumentos.coreset)
    if argumentos.fusion is not None:
        conf = replace(conf, capa_fusion=argumentos.fusion)
    if argumentos.recorte is not None:
        conf = replace(conf, tam_recorte=argumentos.recorte, tam_imagen=argumentos.recorte + 32)
    directorio_salida = os.path.join(conf.dir_salida, "sweeps", argumentos.etiqueta)
    os.makedirs(directorio_salida, exist_ok=True)
    fijar_semilla(conf.semilla)
    registro = configurar_registro(directorio_salida)

    categorias = (MVTEC_CATEGORIAS if argumentos.categorias == "all"
            else tuple(c.strip() for c in argumentos.categorias.split(",")))
    registro.info(f"=== SWEEP post-proceso | config: fusion={conf.capa_fusion} "
                f"coreset={conf.razon_coreset:.0%} | sigmas={SIGMAS} bs={BS} "
                f"| {len(categorias)} categorias | seed={conf.semilla} ===")

    extractor = ExtractorJerarquico(
        conf.backbone, conf.capas, conf.capa_fusion, conf.vecindad, conf.dispositivo)

    campos_b = ["category", "sin_reweight"] + [f"b={b}" for b in BS]
    campos_s = ["category"] + [f"sigma={s:g}" for s in SIGMAS]

    filas_b, filas_s = [], []
    t_total = time.time()
    for categoria in categorias:
        rb, rs = barrer_categoria(conf, categoria, extractor, registro)
        filas_b.append(rb)
        filas_s.append(rs)
        # volcado INCREMENTAL: si el proceso muere, lo ya medido queda en disco
        prom_b = escribir_csv(os.path.join(directorio_salida, "sweep_b.csv"), filas_b, campos_b)
        prom_s = escribir_csv(os.path.join(directorio_salida, "sweep_sigma.csv"), filas_s, campos_s)

    registro.info(f"=== completado en {time.time()-t_total:.1f}s ===")
    registro.info(f"AUROC imagen (medias): " +
                " ".join(f"{k}={prom_b[k]}" for k in campos_b[1:]))
    registro.info(f"AUROC pixel (medias): " +
                " ".join(f"{k}={prom_s[k]}" for k in campos_s[1:]))
    registro.info("--- Wilcoxon pareado vs valores actuales ---")
    wilcoxon_contra_base(filas_b, f"b={B_BASE}", campos_b[1:], "img", registro)
    wilcoxon_contra_base(filas_s, f"sigma={SIGMA_BASE:g}", campos_s[1:], "pix", registro)


if __name__ == "__main__":
    principal()

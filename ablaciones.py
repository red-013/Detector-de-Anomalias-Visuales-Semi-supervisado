'''
Ablations controladas del pipeline PatchCore.

PARA QUE SIRVE
  Medir el efecto de UNA decision de diseno cada vez, dejando TODO lo demas
  igual (misma semilla, mismo coreset, mismo umbral, mismas categorias). Sin eso,
  cualquier diferencia observada seria inatribuible.

ABLATION PRINCIPAL: 2 vs 3 niveles jerarquicos
  El PatchCore canonico usa 2 capas (layer2 + layer3). Nuestra propuesta anade
  layer1 (3 niveles, requisito OE2). Esta ablation responde a la pregunta:
  ?anadir layer1 AYUDA, es NEUTRO, o PERJUDICA?

  Hipotesis a contrastar: layer1 aporta detalle fino (podria subir la AUROC de
  pixel) pero introduce caracteristicas de muy bajo nivel (bordes, color) que en
  objetos con rotacion/posicion libre (screw) pueden anadir ruido y diluir la
  senal semantica de layer2/layer3.

  IMPORTANTE: el resultado se acepta sea cual sea. Si 3 niveles rinde PEOR que 2,
  es un resultado valido y publicable (se ha medido el limite de la propia
  extension). Lo que no seria defendible es asumir 3 > 2 sin comprobarlo.

CONTRASTE ESTADISTICO
  Comparar dos medias sobre 15 categorias no basta: la varianza entre categorias
  es enorme. Se usa el test de WILCOXON de rangos con signo (pareado, no
  parametrico), que es el recomendado para comparar dos metodos sobre multiples
  datasets (Demsar, 2006) porque no asume normalidad de las diferencias y usa el
  emparejamiento natural: cada categoria se evalua con ambas configuraciones.

USO
  python -m modelo_semisupervisado.ablaciones --ablacion niveles
  python -m modelo_semisupervisado.ablaciones --ablacion niveles --categorias bottle,screw
  python -m modelo_semisupervisado.ablaciones --ablacion all
'''
import argparse
import csv
import os
import time
from dataclasses import replace

import numpy as np
import torch
from scipy.stats import wilcoxon

from modelo_semisupervisado.configuracion import obtener_configuracion, fijar_semilla, MVTEC_CATEGORIAS
from modelo_semisupervisado.extractor_caracteristicas import ExtractorJerarquico
from modelo_semisupervisado.principal import ejecutar_categoria, configurar_registro, fila_promedio


# Cada ablation: nombre -> {variante: overrides de Config}
# La PRIMERA variante de cada bloque es la BASE contra la que se compara.
ABLACIONES = {
    # nucleo de la tesis: ?aporta layer1?
    "niveles": {
        "2niveles_canonico": dict(capas=("layer2", "layer3"), capa_fusion="layer2"),
        "3niveles_propuesta": dict(capas=("layer1", "layer2", "layer3"),
                                   capa_fusion="layer2"),
    },
    # ?el banco al 1% se queda corto? (diagnostico: cae AUROC imagen, pixel sano)
    "coreset": {
        "coreset_01": dict(razon_coreset=0.01),
        "coreset_10": dict(razon_coreset=0.10),
    },
    # ?la fusion a 28x28 limita la localizacion? (diagnostico: cae AUROC pixel)
    "fusion": {
        "fusion_28": dict(capa_fusion="layer2"),
        "fusion_56": dict(capa_fusion="layer1"),
    },
    # sensibilidad al backbone
    "backbone": {
        "wide_resnet50_2": dict(backbone="wide_resnet50_2"),
        "resnet50": dict(backbone="resnet50"),
    },
    # ?se pueden COMBINAR las dos palancas que funcionaron (banco y resolucion)?
    #
    # OJO CON EL COSTE: razon_coreset es un % sobre el TOTAL de parches, y a
    # 56x56 hay 4x parches. Por eso:
    #   c10_f28  -> banco = 0.10 * N        (referencia, ya medida)
    #   c025_f56 -> banco = 0.025 * 4N = 0.10 * N   <- MISMO tamano de banco
    #   c10_f56  -> banco = 0.10  * 4N = 0.40 * N   <- x40 vs baseline
    #
    # La comparacion CIENTIFICAMENTE interesante es c10_f28 vs c025_f56: mismo
    # presupuesto de memoria, invertido en DENSIDAD vs en RESOLUCION. Responde a
    # una pregunta de ingenieria real, en vez de "poner todo al maximo".
    #
    # ADVERTENCIA: el greedy es O(m*N). En c10_f56 ambos se multiplican -> ~160x
    # el coste de seleccion del baseline (~90 min sobre las 15 categorias).
    #
    # HIPOTESIS: los efectos NO tienen por que ser aditivos. Ambas palancas
    # atacan el MISMO problema (cobertura del espacio de caracteristicas), asi
    # que es esperable que se solapen.
    "combinada": {
        "c10_f28": dict(razon_coreset=0.10, capa_fusion="layer2"),
        "c025_f56": dict(razon_coreset=0.025, capa_fusion="layer1"),
    },
    "combinada_max": {
        "c10_f28": dict(razon_coreset=0.10, capa_fusion="layer2"),
        "c10_f56": dict(razon_coreset=0.10, capa_fusion="layer1"),
    },
    # ?la RESOLUCION DE ENTRADA explica la brecha con el paper en imagen?
    # (hipotesis viva de la bitacora 8bis: 256->224 pierde defectos pequenos,
    #  p.ej. pill). Se sube manteniendo la proporcion resize/crop del original.
    #
    # DISENO:
    #  - Todas las variantes a capa_fusion="layer2": a 56x56 la entrada 288
    #    pediria ~13 GB de RAM por categoria grande (no cabe). Ademas layer2
    #    aisla el efecto de la entrada sin mezclarlo con el de la fusion.
    #  - OJO: mas resolucion => mas parches => banco mas grande al mismo 1%
    #    (288: 36x36=1296 parches/img vs 784). Y ya sabemos que el tamano del
    #    banco es el factor dominante. Por eso hay variantes *_eq con el ratio
    #    reducido para IGUALAR el banco al de res224 (0.01 * 784/1296, etc.):
    #    si res288 gana pero res288_eq no, la ganancia era del banco, no de la
    #    resolucion; si res288_eq tambien gana, la resolucion aporta por si misma.
    "resolucion": {
        "res224": dict(tam_imagen=256, tam_recorte=224, capa_fusion="layer2"),
        "res288": dict(tam_imagen=320, tam_recorte=288, capa_fusion="layer2"),
        "res288_eq": dict(tam_imagen=320, tam_recorte=288, capa_fusion="layer2",
                          razon_coreset=0.01 * 784 / 1296),
        "res352": dict(tam_imagen=384, tam_recorte=352, capa_fusion="layer2"),
        "res352_eq": dict(tam_imagen=384, tam_recorte=352, capa_fusion="layer2",
                          razon_coreset=0.01 * 784 / 1936),
    },
}

# metricas que se comparan entre variantes
METRICAS = ["auroc_image", "auroc_pixel", "honest_f1", "bestF1_f1"]


# %% Ejecucion de una variante

def ejecutar_variante(configuracion_base, nombre, cambios, categorias, registro):
    '''Corre las categorias con una configuracion concreta y devuelve las filas.'''
    conf = replace(configuracion_base, **cambios)
    # cada variante escribe en su propia carpeta (no se pisan CSV ni logs)
    conf = replace(conf, dir_salida=os.path.join(configuracion_base.dir_salida, "ablations", nombre))
    os.makedirs(conf.dir_salida, exist_ok=True)

    # semilla RESETEADA en cada variante -> la unica diferencia es el override
    fijar_semilla(conf.semilla)

    registro.info(f"--- variante '{nombre}': capas={conf.capas} fusion={conf.capa_fusion} "
                f"coreset={conf.razon_coreset:.0%} backbone={conf.backbone} ---")

    extractor = ExtractorJerarquico(
        conf.backbone, conf.capas, conf.capa_fusion, conf.vecindad, conf.dispositivo)

    filas = []
    for categoria in categorias:
        try:
            filas.append(ejecutar_categoria(conf, categoria, extractor, registro,
                                     con_mapas_calor=False, guardar_banco=False))
        except FileNotFoundError as e:
            registro.warning(f"[{categoria}] SALTADA: {e}")
        except (MemoryError, torch.cuda.OutOfMemoryError) as e:
            # fusion 56x56 cuadruplica los parches: alguna categoria grande puede
            # no caber. Se salta y se DECLARA, en vez de tumbar la ablation entera.
            registro.warning(f"[{categoria}] SALTADA POR MEMORIA ({type(e).__name__}): {e}")
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "memory" in str(e).lower() or "alloc" in str(e).lower():
                registro.warning(f"[{categoria}] SALTADA POR MEMORIA: {e}")
                torch.cuda.empty_cache()
            else:
                raise

    # CSV por variante
    campos = ["category", "auroc_image", "auroc_pixel", "bestF1_f1",
              "bestF1_precision", "bestF1_recall", "honest_f1",
              "honest_precision", "honest_recall", "n_val", "n_coreset"]
    with open(os.path.join(conf.dir_salida, "resultados.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for r in filas:
            w.writerow(r)
        w.writerow(fila_promedio(filas))
    return filas


# %% Contraste estadistico entre variantes

def comparar(resultados, nombre_base, ruta_salida, registro):
    '''Tabla comparativa por categoria + Wilcoxon pareado frente a la base.'''
    nombres = list(resultados.keys())

    # Solo se comparan las categorias presentes en TODAS las variantes, y se
    # indexa POR NOMBRE, no por posicion: si una variante saltase una categoria
    # (p.ej. por memoria en fusion 56x56), comparar por indice enfrentaria
    # categorias distintas y el test seria invalido.
    por_variante = {n: {r["category"]: r for r in resultados[n]} for n in nombres}
    comunes = [c for c in (r["category"] for r in resultados[nombre_base])
              if all(c in por_variante[n] for n in nombres)]
    faltan = {n: sorted(set(por_variante[nombre_base]) - set(por_variante[n]))
              for n in nombres if n != nombre_base}
    for n, ausentes in faltan.items():
        if ausentes:
            registro.warning(f"'{n}' no tiene {ausentes} -> excluidas de la comparativa")

    categorias = comunes
    lineas = []
    if not categorias:
        registro.warning("No hay categorias comunes: comparativa vacia.")
        return
    for metrica in METRICAS:
        lineas.append(f"\n===== {metrica} =====")
        cabecera = f"{'categoria':<13}" + "".join(f"{n:>21}" for n in nombres)
        if len(nombres) == 2:
            cabecera += f"{'delta':>10}"
        lineas.append(cabecera)
        lineas.append("-" * len(cabecera))

        columnas = {n: np.array([por_variante[n][c][metrica] for c in categorias])
                for n in nombres}
        for i, categoria in enumerate(categorias):
            linea = f"{categoria:<13}" + "".join(f"{columnas[n][i]:>21.4f}" for n in nombres)
            if len(nombres) == 2:
                d = columnas[nombres[1]][i] - columnas[nombres[0]][i]
                linea += f"{d:>+10.4f}"
            lineas.append(linea)

        lineas.append("-" * len(cabecera))
        linea = f"{'MEDIA':<13}" + "".join(f"{columnas[n].mean():>21.4f}" for n in nombres)
        if len(nombres) == 2:
            linea += f"{columnas[nombres[1]].mean() - columnas[nombres[0]].mean():>+10.4f}"
        lineas.append(linea)

        # Contraste estadistico pareado frente a la base
        for n in nombres:
            if n == nombre_base:
                continue
            diferencia = columnas[n] - columnas[nombre_base]
            n_mejor = int((diferencia > 0).sum())
            n_peor = int((diferencia < 0).sum())
            if np.allclose(diferencia, 0):
                lineas.append(f"  {n} vs {nombre_base}: identicos (diferencia nula).")
                continue
            try:
                estadistico, p = wilcoxon(columnas[n], columnas[nombre_base])
                sig = "SIGNIFICATIVA (p<0.05)" if p < 0.05 else "NO significativa"
                lineas.append(
                    f"  {n} vs {nombre_base}: gana en {n_mejor}/{len(categorias)} categorias, "
                    f"pierde en {n_peor} | Wilcoxon p={p:.4f} -> diferencia {sig}")
            except ValueError as e:
                lineas.append(f"  {n} vs {nombre_base}: Wilcoxon no aplicable ({e})")

    texto = "\n".join(lineas)
    print(texto)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    registro.info(f"Comparativa guardada en: {ruta_salida}")


# %% Punto de entrada

def principal():
    analizador = argparse.ArgumentParser()
    analizador.add_argument("--ablacion", type=str, default="niveles",
                        help="niveles | coreset | fusion | backbone | all")
    analizador.add_argument("--categorias", type=str, default="all")
    argumentos = analizador.parse_args()

    configuracion_base = obtener_configuracion()
    categorias = (MVTEC_CATEGORIAS if argumentos.categorias == "all"
            else tuple(c.strip() for c in argumentos.categorias.split(",")))

    raiz = os.path.join(configuracion_base.dir_salida, "ablations")
    os.makedirs(raiz, exist_ok=True)
    registro = configurar_registro(raiz)

    todo = list(ABLACIONES) if argumentos.ablacion == "all" else [argumentos.ablacion]
    for abl in todo:
        if abl not in ABLACIONES:
            raise SystemExit(f"ablation desconocida: {abl}. Opciones: {list(ABLACIONES)}")

        variantes = ABLACIONES[abl]
        nombre_base = next(iter(variantes))
        registro.info(f"=== ABLATION '{abl}' | variantes={list(variantes)} "
                    f"| base='{nombre_base}' | {len(categorias)} categorias | seed={configuracion_base.semilla} ===")

        t0 = time.time()
        resultados = {nombre: ejecutar_variante(configuracion_base, nombre, ov, categorias, registro)
                   for nombre, ov in variantes.items()}
        registro.info(f"=== ablation '{abl}' completada en {time.time()-t0:.1f}s ===")

        comparar(resultados, nombre_base, os.path.join(raiz, f"comparativa_{abl}.txt"), registro)


if __name__ == "__main__":
    principal()

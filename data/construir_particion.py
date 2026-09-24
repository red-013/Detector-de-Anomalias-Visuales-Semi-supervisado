'''
CONSTRUCCION DEL CONJUNTO DE DATOS DEL PROYECTO [RESULTADO ESPERADO 4]

Genera `tesis_split/`, una copia REORGANIZADA de MVTec AD y VisA que:

1. Incluye solo las categorias comprendidas en el alcance del proyecto
(se excluyen productos alimenticios, medicos y farmaceuticos);
2. Separa, dentro de cada categoria, las imagenes de entrenamiento de las de
prueba, y dentro del entrenamiento distingue las conformes del subconjunto
reducido de anomalas ETIQUETADAS que exige el regimen semisupervisado.

Uso:

    python construir_particion.py
    python construir_particion.py --particion-visa fewshot
    python construir_particion.py --simulacro
'''
import argparse
import csv
import hashlib
import os
import random
import shutil
import sys
from collections import defaultdict


# CATEGORIAS EXCLUIDAS POR ALCANCE

"""
El alcance del proyecto deja fuera "los productos destinados al consumo
humano, los productos alimenticios, asi como los productos medicos y
farmaceuticos". Se aplica literalmente sobre ambos benchmarks.
"""
MVTEC_EXCLUIDAS = {
    "capsule":   "capsula farmaceutica",
    "pill":      "pildora, producto farmaceutico",
    "hazelnut":  "avellana, producto alimenticio",
}
VISA_EXCLUIDAS = {
    "capsules":   "capsulas farmaceuticas",
    "cashew":     "anacardo, producto alimenticio",
    "chewinggum": "goma de mascar, producto alimenticio",
    "fryum":      "aperitivo, producto alimenticio",
    "macaroni1":  "pasta alimenticia",
    "macaroni2":  "pasta alimenticia",
    "pipe_fryum": "aperitivo, producto alimenticio",
}

MVTEC_CATEGORIAS = ("bottle", "cable", "carpet", "grid", "leather", "metal_nut",
                    "screw", "tile", "toothbrush", "transistor", "wood", "zipper")
VISA_CATEGORIAS = ("candle", "pcb1", "pcb2", "pcb3", "pcb4")


# PROPORCION DEL SUBCONJUNTO ETIQUETADO

"""
VisA distribuye particiones supervisadas oficiales en las que las anomalas de
entrenamiento son exactamente el 10 % de las conformes de entrenamiento
(60/600 en `highshot`, 20/200 en `fewshot`). Para que el protocolo sea el
mismo en los dos benchmarks, MVTec AD usa esa misma regla en vez de una
cantidad arbitraria.
"""
RAZON_ETIQUETADAS = 0.10

"""
Semilla del muestreo. Se fija y declara la semilla para que la seleccion 
pueda rehacerse identica, y no debe depender de haber mirado ningun resultado.
"""
SEMILLA = 13

# TIPO DE DEFECTO RESERVADO (soporte del resultado R6)

"""
R6 mide la deteccion de "tipos de defecto NO representados en el subconjunto
etiquetado". Si el subconjunto etiquetado se muestrease de todos los tipos,
no quedaria ningun tipo no visto y R6 seria imposible de verificar. Por eso,
en cada categoria de MVTec AD se reserva UN tipo de defecto completo, que no
aporta ninguna imagen al entrenamiento y permanece integro en prueba.

El tipo reservado se elige con la semilla fija entre los tipos disponibles
y no a mano: 

Excepcion: `toothbrush` tiene un unico tipo de defecto ("defective"). Ahi no
es posible reservar ninguno sin dejar el subconjunto etiquetado vacio; la
categoria se marca como SIN tipo reservado y queda fuera del analisis de R6.
"""
RESERVAR_TIPO = True

"""
VisA no subdivide sus anomalias por tipo de defecto -ni en la estructura de
carpetas ni en los CSV oficiales-, de modo que en VisA no se puede reservar
un tipo. El analisis de R6 por tipo de defecto queda por tanto acotado a
MVTec AD; es una limitacion de los datasets y no del protocolo.
"""


AQUI = os.path.dirname(os.path.abspath(__file__))      # .../data
RAIZ = os.path.dirname(AQUI)                           # raiz del proyecto
CONJUNTOS_DATOS = os.path.join(AQUI, "datasets")              # corpus e imagenes

RAIZ_MVTEC = os.path.join(CONJUNTOS_DATOS, "mvtec")
RAIZ_VISA = os.path.join(CONJUNTOS_DATOS, "visa")
DESTINO = os.path.join(CONJUNTOS_DATOS, "tesis_split")
DIRECTORIO_CSV = AQUI                                         # los CSV, junto al script

CAMPOS = ("dataset", "categoria", "split", "etiqueta", "tipo_defecto",
          "archivo", "mascara", "origen")


# %% Utilidades de copia y trazabilidad

def copiar(origen, destino, simulacro):
    '''Copia preservando metadatos. Devuelve el numero de bytes copiados.'''
    if simulacro:
        return os.path.getsize(origen)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    shutil.copy2(origen, destino)
    return os.path.getsize(destino)


def sha1_lista(rutas):
    '''Huella de una seleccion, para verificar despues que no ha cambiado.'''
    h = hashlib.sha1()
    for r in sorted(rutas):
        h.update(r.encode("utf-8"))
    return h.hexdigest()[:12]


def ruta_origen(p):
    return os.path.relpath(p, RAIZ).replace(os.sep, "/")


# MVTec AD

# %% Construccion de la particion de MVTec AD

def construir_mvtec(simulacro):
    filas, resumen = [], []
    for categoria in MVTEC_CATEGORIAS:
        generador = random.Random("%s|%s|%d" % ("mvtec", categoria, SEMILLA))
        fuente = os.path.join(RAIZ_MVTEC, categoria)
        ruta_destino = os.path.join(DESTINO, "mvtec", categoria)

        conformes_entrenamiento = sorted(os.listdir(os.path.join(fuente, "train", "good")))
        tipos = sorted(d for d in os.listdir(os.path.join(fuente, "test")) if d != "good")

        """Tipo reservado (ver bloque RESERVAR_TIPO)""" 
        if RESERVAR_TIPO and len(tipos) > 1:
            reservado = generador.choice(tipos)
        else:
            reservado = None
        elegibles = [t for t in tipos if t != reservado]

        """Objetivo: 10 % de las conformes de entrenamiento"""
        objetivo = int(round(RAZON_ETIQUETADAS * len(conformes_entrenamiento)))

        """
        Muestreo estratificado por tipo de defecto: se reparte el cupo a
        partes iguales entre los tipos elegibles y se completa por sorteo.
        Un muestreo simple podria dejar tipos enteros sin representar por
        azar, y entonces la frontera entre "tipo visto" y "tipo no visto",
        que es lo que mide R6, dependeria de la semilla y no del diseño.
        """
        por_tipo = defaultdict(list)
        for t in elegibles:
            por_tipo[t] = sorted(os.listdir(os.path.join(fuente, "test", t)))

        seleccion = []
        if elegibles:
            base = objetivo // len(elegibles)
            for t in elegibles:
                k = min(base, len(por_tipo[t]))
                seleccion += [(t, f) for f in generador.sample(por_tipo[t], k)]
            resto = [(t, f) for t in elegibles for f in por_tipo[t]
                     if (t, f) not in set(seleccion)]
            faltan = min(objetivo - len(seleccion), len(resto))
            if faltan > 0:
                seleccion += generador.sample(resto, faltan)
        conjunto_elegido = set(seleccion)

        """Volcado"""
        n = defaultdict(int)
        for f in conformes_entrenamiento:
            o = os.path.join(fuente, "train", "good", f)
            copiar(o, os.path.join(ruta_destino, "train", "good", f), simulacro)
            filas.append(("mvtec", categoria, "train", "normal", "", "train/good/" + f, "",
                          ruta_origen(o)))
            n["train_good"] += 1

        for t, f in sorted(seleccion):
            nombre = "%s_%s" % (t, f)                     
            o = os.path.join(fuente, "test", t, f)
            copiar(o, os.path.join(ruta_destino, "train", "anomaly", nombre), simulacro)
            om = os.path.join(fuente, "ground_truth", t, f.replace(".png", "_mask.png"))
            mrel = ""
            if os.path.exists(om):
                copiar(om, os.path.join(ruta_destino, "train", "masks", nombre), simulacro)
                mrel = "train/masks/" + nombre
            filas.append(("mvtec", categoria, "train", "anomaly", t, "train/anomaly/" + nombre,
                          mrel, ruta_origen(o)))
            n["train_anom"] += 1

        for f in sorted(os.listdir(os.path.join(fuente, "test", "good"))):
            o = os.path.join(fuente, "test", "good", f)
            copiar(o, os.path.join(ruta_destino, "test", "good", f), simulacro)
            filas.append(("mvtec", categoria, "test", "normal", "", "test/good/" + f, "",
                          ruta_origen(o)))
            n["test_good"] += 1

        for t in tipos:
            for f in sorted(os.listdir(os.path.join(fuente, "test", t))):
                if (t, f) in conjunto_elegido:
                    continue                               # ya esta en entrenamiento
                nombre = "%s_%s" % (t, f)
                o = os.path.join(fuente, "test", t, f)
                copiar(o, os.path.join(ruta_destino, "test", "anomaly", nombre), simulacro)
                om = os.path.join(fuente, "ground_truth", t, f.replace(".png", "_mask.png"))
                mrel = ""
                if os.path.exists(om):
                    copiar(om, os.path.join(ruta_destino, "test", "masks", nombre), simulacro)
                    mrel = "test/masks/" + nombre
                filas.append(("mvtec", categoria, "test", "anomaly", t, "test/anomaly/" + nombre,
                              mrel, ruta_origen(o)))
                n["test_anom"] += 1

        resumen.append(dict(
            dataset="mvtec", categoria=categoria, train_good=n["train_good"],
            train_anom=n["train_anom"], test_good=n["test_good"], test_anom=n["test_anom"],
            ratio=n["train_anom"] / max(1, n["train_good"]),
            tipos=len(tipos), reservado=reservado or "-",
            huella=sha1_lista(["%s/%s" % (t, f) for t, f in seleccion])))
        print("  mvtec/%-11s train %3d good + %2d anom (%.1f %%) | test %3d + %3d | reservado: %s"
              % (categoria, n["train_good"], n["train_anom"],
                 100.0 * n["train_anom"] / max(1, n["train_good"]),
                 n["test_good"], n["test_anom"], reservado or "ninguno"))
    return filas, resumen


# VisA

# %% Construccion de la particion de VisA

def construir_visa(particion, simulacro):
    '''
    VisA NO se re-particiona: se copia tal cual su division oficial.

    Reproducir a mano una particion supervisada sobre VisA seria un error: los
    autores del corpus ya publicaron una, la literatura reporta sobre ella, y
    sustituirla por otra propia romperia la comparabilidad sin ganar nada.
    '''
    ruta_csv = os.path.join(RAIZ_VISA, "split_csv", "2cls_%s.csv" % particion)
    filas, cont = [], defaultdict(lambda: defaultdict(int))
    for r in csv.DictReader(open(ruta_csv, encoding="utf-8")):
        categoria = r["object"]
        if categoria not in VISA_CATEGORIAS:
            continue
        sp, etq = r["split"], ("anomaly" if r["label"] == "anomaly" else "normal")
        sub = "anomaly" if etq == "anomaly" else "good"
        nombre = os.path.basename(r["image"])
        o = os.path.join(RAIZ_VISA, r["image"].replace("/", os.sep))
        d = os.path.join(DESTINO, "visa", categoria, sp, sub, nombre)
        copiar(o, d, simulacro)
        mrel = ""
        if r["mask"].strip():
            om = os.path.join(RAIZ_VISA, r["mask"].replace("/", os.sep))
            mnombre = os.path.basename(r["mask"])
            copiar(om, os.path.join(DESTINO, "visa", categoria, sp, "masks", mnombre), simulacro)
            mrel = "%s/masks/%s" % (sp, mnombre)
        filas.append(("visa", categoria, sp, etq, "", "%s/%s/%s" % (sp, sub, nombre), mrel,
                      ruta_origen(o)))
        cont[categoria]["%s_%s" % (sp, etq)] += 1

    resumen = []
    for categoria in VISA_CATEGORIAS:
        c = cont[categoria]
        resumen.append(dict(
            dataset="visa", categoria=categoria, train_good=c["train_normal"],
            train_anom=c["train_anomaly"], test_good=c["test_normal"],
            test_anom=c["test_anomaly"],
            ratio=c["train_anomaly"] / max(1, c["train_normal"]),
            tipos=0, reservado="-", huella="oficial:2cls_%s" % particion))
        print("  visa/%-12s train %3d good + %2d anom (%.1f %%) | test %3d + %3d | particion oficial"
              % (categoria, c["train_normal"], c["train_anomaly"],
                 100.0 * c["train_anomaly"] / max(1, c["train_normal"]),
                 c["test_normal"], c["test_anomaly"]))
    return filas, resumen


# %% Punto de entrada

def principal():
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--particion-visa", choices=("highshot", "fewshot"), default="highshot")
    analizador.add_argument("--simulacro", action="store_true", help="no copia; solo cuenta")
    argumentos = analizador.parse_args()

    if os.path.exists(DESTINO) and not argumentos.simulacro:
        print("ERROR: %s ya existe. Borrelo o renombrelo antes de regenerar,\n"
              "       para no mezclar dos particiones distintas en la misma carpeta." % DESTINO)
        sys.exit(1)

    print("Construyendo particion del proyecto (semilla %d, VisA=%s)%s\n"
          % (SEMILLA, argumentos.particion_visa, "  [DRY RUN]" if argumentos.simulacro else ""))
    fm, rm = construir_mvtec(argumentos.simulacro)
    print()
    fv, rv = construir_visa(argumentos.particion_visa, argumentos.simulacro)

    if not argumentos.simulacro:
        with open(os.path.join(DIRECTORIO_CSV, "manifiesto.csv"), "w", newline="",
                  encoding="utf-8") as fichero:
            w = csv.writer(fichero)
            w.writerow(CAMPOS)
            w.writerows(fm + fv)
        with open(os.path.join(DIRECTORIO_CSV, "resumen.csv"), "w", newline="",
                  encoding="utf-8") as fichero:
            campos = ("dataset", "categoria", "train_good", "train_anom", "test_good",
                      "test_anom", "ratio", "tipos", "reservado", "huella")
            w = csv.DictWriter(fichero, fieldnames=campos)
            w.writeheader()
            for r in rm + rv:
                r = dict(r)
                r["ratio"] = "%.4f" % r["ratio"]
                w.writerow(r)

    tot = len(fm) + len(fv)
    print("\nTotal: %d imagenes (%d MVTec, %d VisA) en %d categorias."
          % (tot, len(fm), len(fv), len(MVTEC_CATEGORIAS) + len(VISA_CATEGORIAS)))
    if not argumentos.simulacro:
        print("Manifiesto: %s" % os.path.join(DIRECTORIO_CSV, "manifiesto.csv"))


if __name__ == "__main__":
    principal()

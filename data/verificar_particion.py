'''
VERIFICACION DE LA PARTICION DEL PROYECTO [RESULTADO ESPERADO 4]

Rehace las comprobaciones que el documento de la particion declara como
cumplidas. Existe por una razon concreta: una afirmacion del tipo "0
solapamientos entre entrenamiento y prueba" solo vale si cualquiera puede
volver a comprobarla, y el script que GENERA la particion no sirve para eso
-comprobaria su propia logica contra si misma-.

Las cuatro primeras comprobaciones se hacen SOLO con `manifiesto.csv`, sin
necesidad de las imagenes. La quinta (existencia en disco) requiere haber
generado la particion, y se omite sola si la carpeta no esta.

Uso:
    python verificar_particion.py
    python verificar_particion.py --manifiesto otro/manifiesto.csv
    python verificar_particion.py --sin-disco       (omite la comprobacion 5)

Codigo de salida 0 si todo pasa, 1 si alguna comprobacion falla, de modo que
sirva tambien en una comprobacion automatica.
'''
import argparse
import csv
import os
import sys
from collections import defaultdict

AQUI = os.path.dirname(os.path.abspath(__file__))
MANIFIESTO = os.path.join(AQUI, "manifiesto.csv")
RESUMEN = os.path.join(AQUI, "resumen.csv")
CONJUNTOS_DATOS = os.path.join(AQUI, "datasets")
DESTINO = os.path.join(CONJUNTOS_DATOS, "tesis_split")
RAIZ = os.path.dirname(AQUI)


# %% Lectura del manifiesto

def leer(ruta):
    with open(ruta, encoding="utf-8") as fichero:
        return list(csv.DictReader(fichero))


# %% Comprobaciones sobre la particion

def comprobar(nombre, fallos, detalle_max=5):
    '''Imprime el resultado de una comprobacion y devuelve si ha pasado.'''
    ok = not fallos
    print("  [%s] %-52s %s"
          % ("OK" if ok else "!!", nombre,
             "0 incidencias" if ok else "%d incidencias" % len(fallos)))
    for f in fallos[:detalle_max]:
        print("         - %s" % f)
    if len(fallos) > detalle_max:
        print("         ... y %d mas" % (len(fallos) - detalle_max))
    return ok


# %% Punto de entrada

def principal():
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--manifiesto", default=MANIFIESTO)
    analizador.add_argument("--resumen", default=RESUMEN)
    analizador.add_argument("--destino", default=DESTINO)
    analizador.add_argument("--sin-disco", action="store_true",
                    help="omite la comprobacion de existencia de ficheros")
    argumentos = analizador.parse_args()

    filas = leer(argumentos.manifiesto)
    resumen = leer(argumentos.resumen)
    print("Verificacion de la particion: %d filas en el manifiesto, "
          "%d categorias en el resumen.\n" % (len(filas), len(resumen)))

    pasa = []

    # 1. Ninguna imagen de ORIGEN aparece a la vez en train y en test.
    #    Se compara por `origen` y no por `archivo`, porque el nombre cambia al
    #    copiarse (se le antepone el tipo de defecto) y dos rutas distintas
    #    pueden ser la misma imagen del corpus.
    particiones = defaultdict(set)
    for r in filas:
        particiones[r["origen"]].add(r["split"])
    fallos = sorted(o for o, s in particiones.items() if len(s) > 1)
    pasa.append(comprobar("1. Sin solapamiento train/test (por origen)", fallos))

    # 2. Ninguna imagen de origen se copia dos veces.
    veces = defaultdict(int)
    for r in filas:
        veces[r["origen"]] += 1
    fallos = sorted("%s (x%d)" % (o, n) for o, n in veces.items() if n > 1)
    pasa.append(comprobar("2. Sin duplicados de origen", fallos))

    # 3. Toda imagen anomala tiene mascara de segmentacion.
    fallos = [r["archivo"] for r in filas
              if r["etiqueta"] == "anomaly" and not r["mascara"].strip()]
    pasa.append(comprobar("3. Toda anomala tiene mascara", fallos))

    # 4. El tipo reservado no aporta NINGUNA imagen al entrenamiento.
    #    Es la comprobacion que sostiene R6: si un solo fichero del tipo
    #    reservado se colara en train, "tipo no visto" seria falso.
    reservado = {r["categoria"]: r["reservado"] for r in resumen
                 if r["reservado"] not in ("-", "")}
    fallos = [f'{r["categoria"]}/{r["archivo"]}' for r in filas
              if r["split"] == "train" and r["etiqueta"] == "anomaly"
              and r["tipo_defecto"] == reservado.get(r["categoria"])]
    pasa.append(comprobar("4. El tipo reservado no aparece en train", fallos))

    # 4bis. Coherencia entre el manifiesto y el resumen (conteos por categoria).
    conteo = defaultdict(lambda: defaultdict(int))
    for r in filas:
        conteo[r["categoria"]]["%s_%s" % (r["split"], r["etiqueta"])] += 1
    fallos = []
    for r in resumen:
        c, categoria = conteo[r["categoria"]], r["categoria"]
        for columna, clave in (("train_good", "train_normal"),
                           ("train_anom", "train_anomaly"),
                           ("test_good", "test_normal"),
                           ("test_anom", "test_anomaly")):
            if int(r[columna]) != c[clave]:
                fallos.append("%s.%s: resumen %s, manifiesto %d"
                              % (categoria, columna, r[columna], c[clave]))
    pasa.append(comprobar("5. Resumen coherente con el manifiesto", fallos))

    # 6. Todos los ficheros del manifiesto existen en disco.
    if argumentos.sin_disco or not os.path.isdir(argumentos.destino):
        print("  [--] 6. Existencia en disco                              "
              "omitida (no hay particion generada)"
              if not argumentos.sin_disco else
              "  [--] 6. Existencia en disco                              omitida")
    else:
        fallos = []
        for r in filas:
            base = os.path.join(argumentos.destino, r["dataset"], r["categoria"])
            if not os.path.exists(os.path.join(base, r["archivo"])):
                fallos.append(r["archivo"])
            elif r["mascara"].strip() and not os.path.exists(
                    os.path.join(base, r["mascara"])):
                fallos.append(r["mascara"])
        pasa.append(comprobar("6. Todos los ficheros existen en disco", fallos))

    print()
    if all(pasa):
        print("Todas las comprobaciones pasan.")
        return 0
    print("HAY COMPROBACIONES QUE FALLAN. La particion no es valida.")
    return 1


if __name__ == "__main__":
    sys.exit(principal())

# Partición del conjunto de datos (resultado esperado R4)

Protocolo de construcción del conjunto de datos de la tesis, con sus medios de
verificación. Define, para cada categoría comprendida en el alcance, qué
imágenes ve el modelo al entrenar y cuáles quedan reservadas para prueba, en
los dos regímenes de supervisión.

## Contenido

| Fichero | Qué es |
|---|---|
| `construir_particion.py` | genera la partición a partir de los corpus originales |
| `verificar_particion.py` | audita la partición generada; salida 0 si todo pasa |
| `manifiesto.csv` | 9 584 filas, una por imagen, con su procedencia |
| `resumen.csv` | una fila por categoría, con la huella de la selección |
| `datasets/` | corpus e imágenes (~5 GB) — **no versionado** |

Los dos CSV viven fuera de `datasets/` a propósito: son los medios de
verificación de R4 y deben poder consultarse sin descargar las imágenes.
Entre ambos reconstruyen la partición completa, de modo que se puede auditar
el reparto con 900 KB de texto en lugar de 5 GB de fotografías.

## Reproducir la partición

Hay que descargar antes los dos corpus y colocarlos así:

    datasets/mvtec/     MVTec AD  (https://www.mvtec.com/company/research/datasets/mvtec-ad)
    datasets/visa/      VisA      (https://github.com/amazon-science/spot-diff)

VisA debe conservar su carpeta `split_csv/`, que contiene las particiones
oficiales; sin ella no puede reproducirse el protocolo. Si la descarga dejó el
corpus con otro nombre (por ejemplo `archive/`), hay que renombrarlo a `visa`.

    python construir_particion.py                    # genera datasets/tesis_split/
    python verificar_particion.py                # comprueba que es correcta

    python construir_particion.py --simulacro          # ver la composicion sin copiar
    python construir_particion.py --particion-visa fewshot

El script **se niega a escribir sobre una partición existente**: para
regenerarla hay que borrar `datasets/tesis_split/` primero. Es deliberado —
escribir encima dejaría en disco una mezcla de dos particiones que el
manifiesto describiría mal, sin dar ningún error.

## Reproducibilidad

La selección se sortea con **semilla fija (`SEED = 13`)**, declarada en el
código, y nunca a mano. Cada categoría usa un generador propio sembrado con su
nombre, de modo que añadir o quitar categorías del alcance no altera la
selección de las demás.

La columna `huella` de `resumen.csv` es un SHA-1 truncado de la selección de
cada categoría: si dos ejecuciones producen las mismas huellas, la partición es
idéntica. En VisA no se sortea nada — se copia la partición supervisada oficial
del corpus — y la huella lo declara literalmente.

## Documentación completa

Criterios de exclusión por alcance, proporción del subconjunto etiquetado,
elección de la partición de VisA, reserva de un tipo de defecto por categoría,
composición resultante y verificaciones:

    ../PARTICION_DATOS.md

'''
Cifras de referencia de la literatura, recalculadas sobre las categorias del
alcance del proyecto.

POR QUE ESTE MODULO EXISTE. El indicador del resultado R3 exige igualar las
cifras publicadas con un margen de un punto porcentual. Las cifras publicadas
promedian sobre la TOTALIDAD de las categorias de cada corpus -quince en
MVTec AD y doce en VisA- mientras que la delimitacion del alcance reduce la
experimentacion a doce y cinco. Comparar nuestro promedio de doce categorias
contra un promedio publicado de quince no es una comparacion valida: bastaria
con que las categorias excluidas fueran las dificiles para que la diferencia se
debiera al reparto y no al metodo. Aqui se transcriben las tablas DESGLOSADAS
de los articulos y se recalculan los promedios sobre el mismo subconjunto.

TODOS los valores proceden de las tablas de los PDF originales, en
`Referencias_v2/`. Ninguno procede de una busqueda ni de una estimacion, salvo
los explicitamente marcados como tales. Las funciones de comprobacion de este
modulo reproducen los promedios publicados a partir de los valores por
categoria: si una transcripcion fuera erronea, la comprobacion falla.

Ejecutar:  .\\venv\\Scripts\\python.exe -m modelo_semisupervisado.referencias
'''

# PNI (Bae et al., ICCV 2023) - Tabla 1, pagina 6378. MVTec AD por categoria.
# ADVERTENCIA IMPORTANTE sobre que variante recoge esta tabla. Su promedio es
# 99.56 / 98.98, que coincide con la fila "N+P+R" de la Tabla 3 del mismo
# articulo: es decir, la Tabla 1 corresponde al PNI COMPLETO, incluida la red
# de refinamiento R. Nuestra implementacion NO incluye R (se documento como
# fase opcional en el R1). El articulo no publica la variante N+P desglosada
# por categoria, solo su promedio, de modo que la referencia por categoria solo
# puede construirse para el metodo completo. La correccion se aplica mas abajo.
PNI_MVTEC = {                     # categoria: (I-AUROC, P-AUROC) en %
    # Object
    "bottle":     (100.00, 98.87),
    "cable":      (99.76, 99.10),
    "capsule":    (99.72, 99.34),
    "hazelnut":   (100.00, 99.37),
    "metal_nut":  (100.00, 99.29),
    "pill":       (96.99, 99.03),
    "screw":      (99.51, 99.60),
    "toothbrush": (99.72, 99.09),
    "transistor": (100.00, 98.04),
    "zipper":     (99.87, 99.43),
    # Texture
    "carpet":     (100.00, 99.40),
    "grid":       (98.41, 99.20),
    "leather":    (100.00, 99.56),
    "tile":       (100.00, 98.40),
    "wood":       (99.56, 97.04),
}
PNI_MVTEC_PUBLICADO = (99.56, 98.98)      # promedio de las 15, Tabla 1
PNI_MVTEC_OBJETO = (99.55, 99.12)         # sub-promedio de objetos, Tabla 1
PNI_MVTEC_TEXTURA = (99.59, 98.72)        # sub-promedio de texturas, Tabla 1

# Tabla 3 (ablation, pagina 6379): efecto de cada componente sobre el promedio
# de las quince categorias. Solo hay promedios, no desglose por categoria.
PNI_ABLACION = {                  # (I-AUROC, P-AUROC)
    "sin_N_ni_P": (98.92, 98.18),
    "N":         (99.44, 98.62),
    "N+P":       (99.46, 98.80),
    "N+P+R":     (99.56, 98.98),
}

# PNI - Tabla 6, pagina 6381. VisA POR DOMINIO (no por categoria).
# El articulo no desglosa VisA por categoria: agrupa las doce en los tres
# dominios de Zou et al., cuatro categorias cada uno. Esto limita lo que puede
# recalcularse, y el limite no es salvable con los datos publicados.
PNI_VISA_DOMINIO = {              # dominio: (I-AUROC, P-AUROC)
    "instancia unica":      (99.2, 98.1),
    "multiples instancias": (87.9, 99.2),
    "estructura compleja":  (98.5, 99.0),
}
PNI_VISA_PUBLICADO = (95.2, 98.8)         # promedio de las 12, Tabla 6

VISA_DOMINIO_CATEGORIAS = {
    "instancia unica":      ("cashew", "chewinggum", "fryum", "pipe_fryum"),
    "multiples instancias": ("candle", "capsules", "macaroni1", "macaroni2"),
    "estructura compleja":  ("pcb1", "pcb2", "pcb3", "pcb4"),
}

# SoftPatch+ (Wang et al., Pattern Recognition 2025) - Tabla 4, pagina 11.
# VisA por categoria, SEGMENTACION (pixel) unicamente, CON 8 % DE RUIDO.
# Protocolo distinto del nuestro y del de PNI: el articulo inyecta muestras
# anomalas en el conjunto de ENTRENAMIENTO (del 2 % al 40 %) para estudiar la
# robustez al ruido de etiqueta, y su modalidad "No Overlap" ademas retira
# anomalas del conjunto de prueba. Estas cifras NO son una meta; se recogen
# porque son la unica referencia publicada POR CATEGORIA sobre VisA, y por
# tanto lo unico que permite situar nuestras cinco categorias una a una.
SOFTPATCH_VISA_PIXEL = {          # categoria: (No Overlap, Overlap) en %
    "candle":     (98.5, 97.7),
    "capsules":   (98.0, 98.5),
    "cashew":     (99.5, 99.1),
    "chewinggum": (91.6, 91.0),
    "fryum":      (99.4, 99.2),
    "macaroni1":  (98.9, 98.8),
    "macaroni2":  (99.9, 99.6),
    "pcb1":       (97.7, 97.9),
    "pcb2":       (99.9, 98.4),
    "pcb3":       (98.8, 99.8),
    "pcb4":       (99.8, 97.9),
    "pipe_fryum": (99.0, 99.2),
}
SOFTPATCH_VISA_PUBLICADO = (98.4, 98.1)   # promedios de las 12, Tabla 4

# SoftPatch+ - Tabla 1 (pagina 7). MVTec AD por categoria, CLASIFICACION
# (imagen), CON 10 % DE RUIDO. Y Tabla 2 (pagina 8), SEGMENTACION (pixel).
# Las dos modalidades del articulo:
#   "No Overlap" las anomalas inyectadas en entrenamiento se RETIRAN del test.
#   "Overlap"    siguen tambien en el test (el articulo lo llama mas realista).
# Ninguna de las dos coincide con nuestro protocolo, y la diferencia no es
# menor: SoftPatch+ entrena con un 10 % de anomalas NO ETIQUETADAS que trata
# como normales y cuyo efecto trata de anular, mientras que aqui se entrena con
# un 10 % de anomalas ETIQUETADAS que se explotan. Son objetivos opuestos sobre
# la misma proporcion, asi que estas cifras situan el resultado pero no son una
# meta. Ademas usa 256 -> 224 px, frente a nuestros 512 -> 480.
SOFTPATCH_MVTEC_IMAGEN = {        # categoria: (No Overlap, Overlap) en %
    "bottle":     (100.0, 100.0),
    "cable":      (99.0, 88.7),
    "capsule":    (97.1, 98.4),
    "carpet":     (94.0, 96.7),
    "grid":       (99.1, 98.1),
    "hazelnut":   (100.0, 99.1),
    "leather":    (100.0, 100.0),
    "metal_nut":  (99.7, 99.9),
    "pill":       (100.0, 100.0),
    "screw":      (95.7, 95.4),
    "tile":       (99.5, 99.1),
    "toothbrush": (100.0, 99.5),
    "transistor": (99.5, 98.6),
    "wood":       (93.5, 98.5),
    "zipper":     (97.2, 92.8),
}
SOFTPATCH_MVTEC_IMAGEN_PUBLICADO = (98.2, 97.6)   # promedios de las 15, Tabla 1

SOFTPATCH_MVTEC_PIXEL = {         # categoria: (No Overlap, Overlap) en %
    "bottle":     (98.7, 98.4),
    "cable":      (98.4, 98.1),
    "capsule":    (98.6, 98.2),
    "carpet":     (98.6, 98.4),
    "grid":       (99.1, 98.5),
    "hazelnut":   (98.6, 99.1),
    "leather":    (99.3, 98.3),
    "metal_nut":  (96.6, 99.3),
    "pill":       (99.2, 98.6),
    "screw":      (97.1, 97.1),
    "tile":       (96.5, 95.2),
    "toothbrush": (96.7, 94.9),
    "transistor": (96.9, 95.0),
    "wood":       (98.4, 98.7),
    "zipper":     (98.8, 98.7),
}
SOFTPATCH_MVTEC_PIXEL_PUBLICADO = (98.1, 97.8)    # promedios de las 15, Tabla 2

# SoftPatch+ - Tabla 3 (pagina 9). VisA por categoria, CLASIFICACION (imagen),
# CON 8 % DE RUIDO. Complementa la Tabla 4, que ya estaba transcrita abajo.
SOFTPATCH_VISA_IMAGEN = {         # categoria: (No Overlap, Overlap) en %
    "candle":     (73.4, 61.1),
    "capsules":   (98.7, 96.8),
    "cashew":     (94.8, 98.8),
    "chewinggum": (99.0, 94.7),
    "fryum":      (95.6, 95.6),
    "macaroni1":  (98.5, 99.2),
    "macaroni2":  (94.8, 95.0),
    "pcb1":       (78.9, 65.8),
    "pcb2":       (98.5, 97.0),
    "pcb3":       (91.9, 93.4),
    "pcb4":       (96.1, 95.2),
    "pipe_fryum": (99.9, 99.8),
}
SOFTPATCH_VISA_IMAGEN_PUBLICADO = (93.3, 91.0)    # promedios de las 12, Tabla 3

# Categorias del alcance (deben coincidir con configuracion.py).
MVTEC_ALCANCE = ("bottle", "cable", "carpet", "grid", "leather", "metal_nut",
                 "screw", "tile", "toothbrush", "transistor", "wood", "zipper")
VISA_ALCANCE = ("candle", "pcb1", "pcb2", "pcb3", "pcb4")


# %% Comprobacion de las transcripciones

def promedio(tabla, categorias, indice=None):
    '''Promedio simple sobre un subconjunto de categorias.

    Se usa la media aritmetica sin ponderar, que es como los articulos calculan
    sus promedios: cada categoria pesa igual con independencia de cuantas
    imagenes de prueba tenga.
    '''
    valores = [tabla[c] for c in categorias]
    if indice is None:
        n = len(valores[0])
        return tuple(sum(v[i] for v in valores) / len(valores) for i in range(n))
    return sum(v[indice] for v in valores) / len(valores)


def verificar_transcripcion(tol=0.02):
    '''Reproduce los promedios publicados a partir de los valores por categoria.

    Es la comprobacion que hace auditable la transcripcion: si al promediar las
    quince categorias no sale el 99.56 / 98.98 que imprime el articulo, algun
    valor esta mal copiado. La tolerancia cubre el redondeo a dos decimales de
    la tabla original.
    '''
    fallos = []
    for nombre, lista_categorias, esperado in (
        ("MVTec 15 (PNI)", tuple(PNI_MVTEC), PNI_MVTEC_PUBLICADO),
        ("MVTec objeto (PNI)", ("bottle", "cable", "capsule", "hazelnut",
                                "metal_nut", "pill", "screw", "toothbrush",
                                "transistor", "zipper"), PNI_MVTEC_OBJETO),
        ("MVTec textura (PNI)", ("carpet", "grid", "leather", "tile", "wood"),
         PNI_MVTEC_TEXTURA),
    ):
        obt = promedio(PNI_MVTEC, lista_categorias)
        for i, m in enumerate(("imagen", "pixel")):
            if abs(obt[i] - esperado[i]) > tol:
                fallos.append("%s %s: calculado %.3f, publicado %.2f"
                              % (nombre, m, obt[i], esperado[i]))

    # VisA de PNI: el promedio de los tres dominios debe dar el publicado.
    d = PNI_VISA_DOMINIO
    for i, m in enumerate(("imagen", "pixel")):
        obt = sum(v[i] for v in d.values()) / 3.0
        if abs(obt - PNI_VISA_PUBLICADO[i]) > 0.05:
            fallos.append("VisA dominios %s: calculado %.3f, publicado %.2f"
                          % (m, obt, PNI_VISA_PUBLICADO[i]))

    # SoftPatch+: las CUATRO tablas, en sus dos modalidades. La tolerancia es
    # mayor que en PNI porque el articulo tabula con tres decimales en fraccion
    # (0.982), de modo que el promedio publicado arrastra el redondeo de quince
    # o doce valores truncados.
    for nombre, tabla, pub in (
        ("MVTec imagen", SOFTPATCH_MVTEC_IMAGEN, SOFTPATCH_MVTEC_IMAGEN_PUBLICADO),
        ("MVTec pixel", SOFTPATCH_MVTEC_PIXEL, SOFTPATCH_MVTEC_PIXEL_PUBLICADO),
        ("VisA imagen", SOFTPATCH_VISA_IMAGEN, SOFTPATCH_VISA_IMAGEN_PUBLICADO),
        ("VisA pixel", SOFTPATCH_VISA_PIXEL, SOFTPATCH_VISA_PUBLICADO),
    ):
        for i, modo in enumerate(("No Overlap", "Overlap")):
            obt = promedio(tabla, tuple(tabla), indice=i)
            if abs(obt - pub[i]) > 0.11:
                fallos.append("SoftPatch+ %s %s: calculado %.3f, publicado %.2f"
                              % (nombre, modo, obt, pub[i]))
    return fallos


# %% Consulta de cifras de referencia

def referencia_mvtec():
    '''Referencia de MVTec AD recalculada sobre las doce categorias del alcance.

    Devuelve un dict con dos entradas:
      "completo"  PNI tal y como lo publica la Tabla 1, es decir CON la red de
                  refinamiento R.
      "sin_R"     estimacion de la variante N+P, que es la que corresponde a
                  nuestra implementacion. Se obtiene restando el efecto que la
                  Tabla 3 atribuye a R sobre el promedio de las quince
                  categorias (+0.10 de imagen y +0.18 de pixel). ES UNA
                  ESTIMACION: el articulo no publica N+P por categoria, de modo
                  que se asume que el efecto de R se reparte por igual entre
                  ellas, cosa que el propio articulo desmiente al senalar que el
                  refinamiento es mas efectivo en texturas. Debe reportarse como
                  aproximada.
    '''
    comp = promedio(PNI_MVTEC, MVTEC_ALCANCE)
    dR = (PNI_ABLACION["N+P+R"][0] - PNI_ABLACION["N+P"][0],
          PNI_ABLACION["N+P+R"][1] - PNI_ABLACION["N+P"][1])
    return {
        "completo": comp,
        "sin_R": (comp[0] - dR[0], comp[1] - dR[1]),
        "delta_R": dR,
        "publicado_15": PNI_MVTEC_PUBLICADO,
        "n_categorias": len(MVTEC_ALCANCE),
    }


def referencia_visa():
    '''Referencia de VisA sobre las cinco categorias del alcance.

    Solo es exacta para el subconjunto de estructura compleja. PNI publica VisA
    agrupada por dominio, y las cuatro categorias del dominio de estructura
    compleja son exactamente las cuatro placas de circuito impreso que quedan en
    el alcance, de modo que su cifra se aplica sin transformacion alguna.

    `candle` no tiene referencia individual publicada: pertenece a un dominio de
    cuatro categorias del que las otras tres quedan excluidas por alcance, y de
    una media de cuatro valores no puede extraerse uno de ellos. NO se estima
    asignandole el promedio de su dominio (87.9 de imagen): ese promedio esta
    arrastrado por `capsules` y `macaroni2`, que la literatura senala como las
    mas dificiles, de modo que usarlo como referencia de `candle` produciria una
    meta artificialmente baja, es decir, sesgada a nuestro favor.
    '''
    return {
        "estructura_compleja": PNI_VISA_DOMINIO["estructura compleja"],
        "categorias_exactas": ("pcb1", "pcb2", "pcb3", "pcb4"),
        "sin_referencia": ("candle",),
        "softpatch_pixel_alcance": {
            "no_overlap": promedio(SOFTPATCH_VISA_PIXEL, VISA_ALCANCE, indice=0),
            "overlap": promedio(SOFTPATCH_VISA_PIXEL, VISA_ALCANCE, indice=1),
        },
        "publicado_12": PNI_VISA_PUBLICADO,
    }


# %% Informe por consola

def _informe():
    print("=" * 74)
    print("VERIFICACION DE LA TRANSCRIPCION")
    print("=" * 74)
    fallos = verificar_transcripcion()
    if fallos:
        for f in fallos:
            print("  FALLO:", f)
    else:
        print("  Los promedios calculados reproducen los publicados. OK.")

    m = referencia_mvtec()
    print()
    print("=" * 74)
    print("MVTec AD - referencia recalculada sobre %d categorias" % m["n_categorias"])
    print("=" * 74)
    print("  publicado sobre las 15 (PNI completo) : %.2f img / %.2f pix"
          % PNI_MVTEC_PUBLICADO)
    print("  recalculado sobre las 12 del alcance  : %.2f img / %.2f pix"
          % m["completo"])
    print("  efecto de excluir capsule/hazelnut/pill: %+.2f img / %+.2f pix"
          % (m["completo"][0] - PNI_MVTEC_PUBLICADO[0],
             m["completo"][1] - PNI_MVTEC_PUBLICADO[1]))
    print()
    print("  Nuestra implementacion no incluye la red de refinamiento R.")
    print("  referencia N+P estimada sobre las 12   : %.2f img / %.2f pix  (APROXIMADA)"
          % m["sin_R"])
    print("  margen de 1 punto -> objetivo minimo    : %.2f img / %.2f pix"
          % (m["sin_R"][0] - 1.0, m["sin_R"][1] - 1.0))

    v = referencia_visa()
    print()
    print("=" * 74)
    print("VisA - referencia sobre las 5 categorias del alcance")
    print("=" * 74)
    print("  publicado sobre las 12 (PNI)           : %.2f img / %.2f pix"
          % PNI_VISA_PUBLICADO)
    print("  pcb1-4 (dominio completo, EXACTO)      : %.2f img / %.2f pix"
          % v["estructura_compleja"])
    print("  candle                                 : sin referencia individual publicada")
    print("  margen de 1 punto sobre pcb1-4         : %.2f img / %.2f pix"
          % (v["estructura_compleja"][0] - 1.0, v["estructura_compleja"][1] - 1.0))
    print()
    sp = v["softpatch_pixel_alcance"]
    print("  SoftPatch+ pixel sobre las 5 del alcance (8 %% de ruido, otro protocolo):")
    print("    No Overlap %.2f  |  Overlap %.2f" % (sp["no_overlap"], sp["overlap"]))
    print()
    print("  LIMITACION: PNI publica VisA agrupada por dominio, no por categoria.")
    print("  De una media de cuatro valores no puede extraerse el de `candle`, y")
    print("  asignarle el promedio de su dominio (87.9) daria una meta")
    print("  artificialmente baja. Se reporta sin referencia.")


if __name__ == "__main__":
    _informe()

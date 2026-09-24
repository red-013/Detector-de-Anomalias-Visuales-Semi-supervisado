'''
Modulo (a): carga y preprocesamiento de MVTec AD.

Claves de diseno:
  - Split estructuralmente one-class: train/ solo tiene 'good', asi que el
    Dataset de train NUNCA puede devolver una imagen defectuosa. Cumple el
    requisito de entrenamiento estricto one-class a nivel de datos.
  - Misma transformacion geometrica en imagen y mascara -> el defecto sigue
    alineado pixel a pixel tras resize/crop.
  - Mascara con interpolacion NEAREST y sin normalizar: es binaria, no se
    deben inventar valores intermedios.
  - Todo determinista (shuffle=False) -> banco reproducible.
'''
import os
from glob import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

IMAGENET_MEDIA = [0.485, 0.456, 0.406]
IMAGENET_DESV = [0.229, 0.224, 0.225]


# %% Transformaciones de imagen

def construir_transformaciones(tam_imagen=256, tam_recorte=224, deformar=False):
    '''Transformacion geometrica, comun a imagen y mascara.

    squash=False (MVTec): Resize del lado corto + CenterCrop. MVTec es cuadrado,
        asi que el recorte apenas descarta contenido.

    squash=True (VisA): Resize DIRECTO a (tam_recorte, tam_recorte), sin recorte.
        VisA es APAISADO (~1500x1000). Con el esquema de MVTec, un Resize(512)
        dejaria ~768x512 y el CenterCrop(480) descartaria cerca del 40% de la
        imagen por los lados, SIN emitir ningun error: si el defecto cae en esa
        franja, simplemente desaparece y las metricas bajan sin causa aparente.
        Deformar la relacion de aspecto es preferible a perder region anotada.
    '''
    if deformar:
        geom_imagen = [T.Resize((tam_recorte, tam_recorte),
                             interpolation=T.InterpolationMode.BILINEAR)]
        geom_mascara = [T.Resize((tam_recorte, tam_recorte),
                             interpolation=T.InterpolationMode.NEAREST)]
    else:
        geom_imagen = [T.Resize(tam_imagen, interpolation=T.InterpolationMode.BILINEAR),
                    T.CenterCrop(tam_recorte)]
        geom_mascara = [T.Resize(tam_imagen, interpolation=T.InterpolationMode.NEAREST),
                    T.CenterCrop(tam_recorte)]

    transf_imagen = T.Compose(geom_imagen + [
        T.ToTensor(),                                   # [0,1], (C,H,W)
        T.Normalize(mean=IMAGENET_MEDIA, std=IMAGENET_DESV),
    ])
    # La mascara se queda en PIL: SOLO se le aplica la geometria. La conversion
    # a tensor la hace cargar_mascara(), por el motivo que se explica alli.
    transf_mascara = T.Compose(geom_mascara)
    return transf_imagen, transf_mascara


# %% Carga de mascaras de segmentacion

def cargar_mascara(ruta, transf_mascara, hw):
    '''Carga una mascara de segmentacion y la binariza sin suponer su escala.

    AQUI HABIA UN FALLO SILENCIOSO. La version anterior hacia
    `ToTensor()` y luego `mask > 0.5`. ToTensor divide entre 255 los enteros de
    una imagen en modo "L", asi que:

      - MVTec AD guarda las mascaras como {0, 255} -> 255/255 = 1.0 > 0.5  OK.
      - VisA NO. Sus mascaras son MAPAS DE ETIQUETAS con un entero pequeno por
        instancia de defecto: los valores medidos van de 1 a 7. Tras ToTensor
        eso queda en 1/255 = 0.0039, que NO supera 0.5, de modo que TODAS las
        mascaras de VisA salian completamente a cero. El AUROC de pixel se
        habria calculado contra un ground truth vacio (o habria reventado por
        tener una sola clase), sin ningun mensaje de error.

    La regla correcta no depende de la escala: cualquier pixel distinto de cero
    pertenece a un defecto. La geometria se aplica en PIL con interpolacion
    NEAREST, asi que los valores siguen siendo los enteros originales y no
    aparecen valores intermedios inventados por la interpolacion.
    '''
    if ruta is None or not os.path.exists(ruta):
        return torch.zeros(1, *hw)
    m = transf_mascara(Image.open(ruta).convert("L"))
    arreglo = np.array(m)
    return torch.from_numpy((arreglo > 0).astype("float32")).unsqueeze(0)


# %% Conjuntos de datos originales

class ConjuntoMVTec(Dataset):
    '''
    split='train' -> solo good; __getitem__ devuelve: imagen
    split='test'  -> good+defectos; __getitem__ devuelve: (imagen, label, mascara)
    '''

    def __init__(self, raiz, categoria, particion, tam_imagen=256, tam_recorte=224):
        self.particion = particion
        self.transf_imagen, self.transf_mascara = construir_transformaciones(tam_imagen, tam_recorte,
                                                    deformar=False)
        directorio_categoria = os.path.join(raiz, categoria)
        if not os.path.isdir(directorio_categoria):
            raise FileNotFoundError(
                f"No existe la categoria en: {directorio_categoria}\n"
                f"Descomprime MVTec AD de forma que quede {directorio_categoria}\\train\\good\\*.png"
            )
        self.muestras = []  # (ruta_img, label, ruta_mask|None)

        if particion == "train":
            for p in sorted(glob(os.path.join(directorio_categoria, "train", "good", "*.png"))):
                self.muestras.append((p, 0, None))
        else:
            directorio_prueba = os.path.join(directorio_categoria, "test")
            directorio_gt = os.path.join(directorio_categoria, "ground_truth")
            for defecto in sorted(os.listdir(directorio_prueba)):
                for p in sorted(glob(os.path.join(directorio_prueba, defecto, "*.png"))):
                    if defecto == "good":
                        self.muestras.append((p, 0, None))
                    else:
                        nombre = os.path.splitext(os.path.basename(p))[0]
                        mascara = os.path.join(directorio_gt, defecto, f"{nombre}_mask.png")
                        self.muestras.append((p, 1, mascara))

        if len(self.muestras) == 0:
            raise RuntimeError(f"0 imagenes encontradas para {categoria}/{particion}.")

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, i):
        ruta, etiqueta, ruta_mascara = self.muestras[i]
        imagen = self.transf_imagen(Image.open(ruta).convert("RGB"))

        if self.particion == "train":
            return imagen

        mascara = cargar_mascara(ruta_mascara, self.transf_mascara, (imagen.shape[1], imagen.shape[2]))
        return imagen, etiqueta, mascara


class ConjuntoVisA(Dataset):
    '''VisA (Zou et al., ECCV 2022), hermano de ConjuntoMVTec.

    Tres diferencias con MVTec que hay que respetar, todas ellas fuente de
    fallos silenciosos si se ignoran:

      1) LAYOUT. VisA no separa train/test por carpetas, sino por un CSV:
         split_csv/1cls.csv, con columnas (object, split, label, image, mask).
         Se usa el split OFICIAL: inventar uno propio destruiria la
         comparabilidad con la literatura, que es la razon de anadir VisA.

      2) FORMATO. Las imagenes son JPG y las mascaras PNG. Un glob de "*.png"
         sobre las imagenes devolveria cero resultados.

      3) RELACION DE ASPECTO. Las imagenes son apaisadas; la transformacion usa
         squash=True (ver construir_transformaciones) para no recortar los laterales.

    Si el CSV no esta, se cae a un recorrido por directorios
    (Data/Images/Normal | Anomaly, Data/Masks/Anomaly) y se avisa, porque en ese
    caso el split ya no es el oficial.
    '''

    EXTENSIONES_IMAGEN = ("*.JPG", "*.jpg", "*.png")

    def __init__(self, raiz, categoria, particion, tam_imagen=512, tam_recorte=480):
        self.particion = particion
        self.transf_imagen, self.transf_mascara = construir_transformaciones(tam_imagen, tam_recorte,
                                                     deformar=True)
        directorio_categoria = os.path.join(raiz, categoria)
        if not os.path.isdir(directorio_categoria):
            raise FileNotFoundError(
                f"No existe la categoria en: {directorio_categoria}\n"
                f"Descomprime VisA de forma que quede {directorio_categoria}\\Data\\Images\\"
            )
        self.muestras = []
        ruta_csv = os.path.join(raiz, "split_csv", "1cls.csv")
        if os.path.exists(ruta_csv):
            self._cargar_desde_csv(raiz, ruta_csv, categoria, particion)
        else:
            print(f"[AVISO] No se encontro {ruta_csv}: se usa un recorrido por "
                  f"directorios y el split NO es el oficial de VisA.")
            self._cargar_desde_directorios(directorio_categoria, particion)

        if len(self.muestras) == 0:
            raise RuntimeError(f"0 imagenes encontradas para {categoria}/{particion}.")

    def _cargar_desde_csv(self, raiz, ruta_csv, categoria, particion):
        import csv as _csv
        esperados = "train" if particion == "train" else "test"
        with open(ruta_csv, newline="", encoding="utf-8") as f:
            for fila in _csv.DictReader(f):
                if fila["object"] != categoria or fila["split"] != esperados:
                    continue
                etiqueta = 0 if fila["label"].strip().lower() == "normal" else 1
                if particion == "train" and etiqueta != 0:
                    continue                      # el banco solo ve good
                imagen = os.path.join(raiz, fila["image"])
                masc = fila.get("mask") or ""
                masc = os.path.join(raiz, masc) if masc.strip() else None
                self.muestras.append((imagen, etiqueta, masc))

    def _cargar_desde_directorios(self, directorio_categoria, particion):
        base = os.path.join(directorio_categoria, "Data")
        norma = os.path.join(base, "Images", "Normal")
        anomalas = os.path.join(base, "Images", "Anomaly")
        dir_masc = os.path.join(base, "Masks", "Anomaly")

        def _buscar_ficheros(d):
            salida = []
            for ext in self.EXTENSIONES_IMAGEN:
                salida += glob(os.path.join(d, ext))
            return sorted(set(salida))

        conformes = _buscar_ficheros(norma)
        if particion == "train":
            self.muestras = [(p, 0, None) for p in conformes[: int(len(conformes) * 0.8)]]
            return
        for p in conformes[int(len(conformes) * 0.8):]:
            self.muestras.append((p, 0, None))
        for p in _buscar_ficheros(anomalas):
            nombre = os.path.splitext(os.path.basename(p))[0]
            self.muestras.append((p, 1, os.path.join(dir_masc, f"{nombre}.png")))

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, i):
        ruta, etiqueta, ruta_mascara = self.muestras[i]
        imagen = self.transf_imagen(Image.open(ruta).convert("RGB"))
        if self.particion == "train":
            return imagen
        mascara = cargar_mascara(ruta_mascara, self.transf_mascara, (imagen.shape[1], imagen.shape[2]))
        return imagen, etiqueta, mascara


# %% Particion semisupervisada

def apartar_conformes_validacion(n_entrenamiento, razon_validacion=0.10, semilla=0):
    '''
    Aparta un subconjunto de good de train para calibrar el umbral honesto
    (modulo e). Estas imagenes NO entran al banco -> sin data leakage.
    Devuelve (indices_banco, indices_val).
    '''
    g = torch.Generator().manual_seed(semilla)
    permutacion = torch.randperm(n_entrenamiento, generator=g).tolist()
    n_val = int(n_entrenamiento * razon_validacion)
    return permutacion[n_val:], permutacion[:n_val]


class ConjuntoParticionTesis(Dataset):
    '''Lee la particion propia del proyecto (`data/datasets/tesis_split/`, R4).

    A diferencia de ConjuntoMVTec y ConjuntoVisA, que leen los corpus originales
    con su estructura de una sola clase, esta clase lee la particion ya
    resuelta y por tanto NO decide nada: que imagen es de entrenamiento y cual
    de prueba viene fijado por R4. Esa separacion es deliberada -si el reparto
    se recalculase aqui, cambiar un hiperparametro del modelo podria cambiar
    tambien el conjunto de datos y las comparaciones dejarian de ser validas.

    Estructura leida:
        <raiz>/<dataset>/<categoria>/train/good/*
        <raiz>/<dataset>/<categoria>/train/anomaly/*   + train/masks/*
        <raiz>/<dataset>/<categoria>/test/good/*
        <raiz>/<dataset>/<categoria>/test/anomaly/*    + test/masks/*

    `particion` admite tres valores:
        "train_good"    solo las conformes de entrenamiento (regimen one-class)
        "train_anomaly" solo las anomalas etiquetadas (regimen semisupervisado)
        "test"          el conjunto de prueba completo

    __getitem__ devuelve SIEMPRE la terna (imagen, etiqueta, mascara), tambien
    en entrenamiento. En el resto del pipeline las conformes se piden con
    `solo_imagen=True` para conservar la firma que espera BancoDeMemoria.fit.
    '''

    SUBDIRECTORIO = {"train_good": ("train", "good"),
              "train_anomaly": ("train", "anomaly"),
              "test": ("test", None)}

    def __init__(self, raiz, conjunto_datos, categoria, particion,
                 tam_imagen=512, tam_recorte=480, deformar=None, solo_imagen=False):
        if particion not in self.SUBDIRECTORIO:
            raise ValueError("split invalido: %r" % particion)
        self.particion = particion
        self.solo_imagen = solo_imagen
        # VisA sigue siendo apaisado tambien en la particion: la geometria es
        # una propiedad del corpus, no del reparto.
        if deformar is None:
            deformar = (conjunto_datos == "visa")
        self.transf_imagen, self.transf_mascara = construir_transformaciones(tam_imagen, tam_recorte, deformar)

        directorio_categoria = os.path.join(raiz, conjunto_datos, categoria)
        if not os.path.isdir(directorio_categoria):
            raise FileNotFoundError(
                "No existe %s.\nGenera la particion con:\n"
                "    .\\venv\\Scripts\\python.exe data\\construir_particion.py" % directorio_categoria)

        self.muestras = []                       # (ruta_img, label, ruta_mask|None)
        sp, sub = self.SUBDIRECTORIO[particion]
        subdirectorios = [sub] if sub else ["good", "anomaly"]
        for s in subdirectorios:
            d = os.path.join(directorio_categoria, sp, s)
            if not os.path.isdir(d):
                continue
            for p in sorted(glob(os.path.join(d, "*"))):
                if s == "good":
                    self.muestras.append((p, 0, None))
                else:
                    # La mascara conserva el nombre base de la imagen salvo la
                    # extension: VisA guarda las imagenes en JPG y las mascaras
                    # en PNG, y build_split respeta ambos nombres de origen.
                    m = self._buscar_mascara(os.path.join(directorio_categoria, sp, "masks"), p)
                    self.muestras.append((p, 1, m))

        if not self.muestras:
            raise RuntimeError("0 imagenes en %s/%s" % (directorio_categoria, particion))

    @staticmethod
    def _buscar_mascara(dir_mascaras, ruta_imagen):
        base = os.path.splitext(os.path.basename(ruta_imagen))[0]
        for ext in (".png", ".PNG", ".jpg", ".JPG"):
            c = os.path.join(dir_mascaras, base + ext)
            if os.path.exists(c):
                return c
        # Sin mascara la imagen no puede aportar parches anomalos ni evaluarse a
        # nivel de pixel. No se silencia: R4 verifico que todas la tienen, asi
        # que su ausencia significa que la particion esta incompleta.
        raise FileNotFoundError("Falta la mascara de %s en %s" % (ruta_imagen, dir_mascaras))

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, i):
        ruta, etiqueta, ruta_mascara = self.muestras[i]
        imagen = self.transf_imagen(Image.open(ruta).convert("RGB"))
        if self.solo_imagen:
            return imagen
        mascara = cargar_mascara(ruta_mascara, self.transf_mascara, (imagen.shape[1], imagen.shape[2]))
        return imagen, etiqueta, mascara


# %% Construccion de cargadores

def construir_cargadores_particion(conf, categoria):
    '''DataLoaders sobre la particion de R4.

    Devuelve cuatro cargadores:
      cargador_entrenamiento : conformes del banco (sin las apartadas para validacion)
      val_loader   : conformes apartadas, para calibrar el umbral sin fuga
      cargador_anomalas  : anomalas etiquetadas (vacio en regimen one-class)
      cargador_prueba  : conjunto de prueba

    Las conformes se subdividen en banco y validacion con la misma funcion y la
    misma semilla que usaba el pipeline anterior, de modo que la unica variable
    que cambia entre los dos regimenes sea la presencia de `cargador_anomalas`.
    '''
    conj = conf.conjunto_datos
    conformes_completo = ConjuntoParticionTesis(conf.raiz_particion, conj, categoria, "train_good",
                                  conf.tam_imagen, conf.tam_recorte, solo_imagen=True)
    conjunto_prueba = ConjuntoParticionTesis(conf.raiz_particion, conj, categoria, "test",
                                conf.tam_imagen, conf.tam_recorte)

    indices_banco, ind_validacion = apartar_conformes_validacion(
        len(conformes_completo), conf.razon_validacion, conf.semilla)
    cargador_entrenamiento = DataLoader(Subset(conformes_completo, indices_banco),
                                        batch_size=conf.tam_lote,
                              shuffle=False, num_workers=conf.n_trabajadores)
    cargador_validacion = DataLoader(Subset(conformes_completo, ind_validacion),
                                     batch_size=conf.tam_lote,
                            shuffle=False, num_workers=conf.n_trabajadores)
    cargador_prueba = DataLoader(conjunto_prueba, batch_size=1, shuffle=False,
                             num_workers=conf.n_trabajadores)

    cargador_anomalas = None
    if conf.regimen == "semi":
        conjunto_anomalas = ConjuntoParticionTesis(conf.raiz_particion, conj, categoria, "train_anomaly",
                                    conf.tam_imagen, conf.tam_recorte)
        cargador_anomalas = DataLoader(conjunto_anomalas, batch_size=conf.tam_lote, shuffle=False,
                                 num_workers=conf.n_trabajadores)
    return cargador_entrenamiento, cargador_validacion, cargador_anomalas, cargador_prueba


def construir_cargadores(conf, categoria):
    '''
    Construye los DataLoaders de una categoria a partir de la Config.
      - cargador_entrenamiento: solo las good del BANCO (sin las de validacion).
      - val_loader:   good apartadas para el umbral (imagenes sueltas).
      - cargador_prueba:  good+defectos (batch=1, cada imagen -> su mapa).
    '''
    ClaseConjunto = ConjuntoVisA if getattr(conf, "conjunto_datos", "mvtec") == "visa" else ConjuntoMVTec
    entrenamiento_completo = ClaseConjunto(conf.raiz_datos, categoria, "train",
                    conf.tam_imagen, conf.tam_recorte)
    conjunto_prueba = ClaseConjunto(conf.raiz_datos, categoria, "test",
                 conf.tam_imagen, conf.tam_recorte)

    indices_banco, ind_validacion = apartar_conformes_validacion(
        len(entrenamiento_completo), conf.razon_validacion, conf.semilla)
    conjunto_banco = Subset(entrenamiento_completo, indices_banco)
    conjunto_validacion = Subset(entrenamiento_completo, ind_validacion)

    cargador_entrenamiento = DataLoader(conjunto_banco, batch_size=conf.tam_lote, shuffle=False,
                              num_workers=conf.n_trabajadores)
    cargador_validacion = DataLoader(conjunto_validacion, batch_size=conf.tam_lote, shuffle=False,
                            num_workers=conf.n_trabajadores)
    cargador_prueba = DataLoader(conjunto_prueba, batch_size=1, shuffle=False,
                             num_workers=conf.n_trabajadores)
    return cargador_entrenamiento, cargador_validacion, cargador_prueba


if __name__ == "__main__":
    # Prueba rapida de humo (ejecuta cuando ya tengas el dataset descargado):
    #   python -m modelo_semisupervisado.conjuntos_datos
    from modelo_semisupervisado.configuracion import obtener_configuracion
    conf = obtener_configuracion()
    ent, val, pru = construir_cargadores(conf, "bottle")
    print(f"[OK] banco={len(ent.dataset)}  val={len(val.dataset)}  test={len(pru.dataset)}")
    imagen = next(iter(ent))
    print("batch train:", imagen.shape)          # (B, 3, 224, 224)
    x, y, m = next(iter(pru))
    print("test img:", x.shape, "label:", y.item(), "mask:", m.shape)

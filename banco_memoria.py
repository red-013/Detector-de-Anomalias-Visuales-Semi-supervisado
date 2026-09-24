'''
Modulo (c): banco de memoria + submuestreo por greedy coreset.

Idea: memorizamos la normalidad a nivel de parche. El banco M son TODOS los
vectores de parche de las imagenes good. Consultar el vecino mas cercano contra
todo M es caro y redundante (muchos parches de fondo casi identicos), asi que
lo reducimos con un coreset que CUBRE geometricamente el espacio de features.

Por que coreset y no muestreo aleatorio:
  El aleatorio sobre-representa las zonas densas (fondos repetidos) y pierde los
  parches raros pero validos (bordes de la pieza), que son justo los que generan
  falsos positivos si faltan. El coreset resuelve un problema minimax (k-center):

      min_C  max_{x in M}  min_{c in C} ||x - c||

  o sea: elige C para que ningun punto de M quede lejos de su representante.

Greedy k-center (2-aproximado):
  1) empieza con un punto (aleatorio, semilla fija).
  2) para cada punto guarda su distancia al conjunto ya elegido (min_dist).
  3) anade el punto con MAYOR min_dist (el mas "descubierto").
  4) actualiza min_dist con el recien anadido. Repite hasta llenar m.

Dos trucos para que sea viable en millones de vectores:
  - Proyeccion aleatoria Johnson-Lindenstrauss: reduce la dimension (p.ej. 1792
    -> 128) preservando aprox. las distancias (lema JL). Se usa SOLO para elegir;
    el banco final guarda los vectores ORIGINALES completos.
  - Distancias al cuadrado (sin sqrt): sqrt es monotona, no cambia el argmax/argmin.
'''
import time
import torch
from tqdm import tqdm


# %% Proyeccion y seleccion de coreset

@torch.no_grad()
def _proyectar_jl(caracteristicas, dim_proyeccion, dispositivo, semilla):
    '''Proyeccion aleatoria gaussiana (JL) por lotes -> (N, dim_proyeccion) en device.

    Solo se usa para calcular distancias en la seleccion greedy. La escala global
    de la proyeccion es irrelevante (solo comparamos distancias entre si), asi que
    basta una matriz gaussiana; el lema JL garantiza que las distancias relativas
    se conservan de forma aproximada.
    '''
    N, C = caracteristicas.shape
    if not dim_proyeccion or dim_proyeccion >= C:
        return caracteristicas.to(dispositivo)
    g = torch.Generator().manual_seed(semilla)              # CPU generator
    proy = torch.randn(C, dim_proyeccion, generator=g)   # (C, d')
    salida = torch.empty(N, dim_proyeccion, device=dispositivo)
    tam_bloque = 4096
    for i in range(0, N, tam_bloque):                            # proyecta por lotes
        # .float(): el pool de parches se guarda en float16 para que quepa en
        # RAM, pero el producto se hace en float32 (matmul no acepta mezclar
        # dtypes y en fp16 la suma de 1792 terminos pierde precision).
        salida[i:i + tam_bloque] = (caracteristicas[i:i + tam_bloque].float() @ proy).to(dispositivo)
    return salida


@torch.no_grad()
def coreset_voraz(caracteristicas, razon, dim_proyeccion, dispositivo, semilla=0, n_seleccion=None):
    '''
    caracteristicas : (N, C) en CPU  -> conjunto candidato
    razon    : fraccion a seleccionar (ignorada si se pasa n_seleccion)
    n_seleccion : numero EXACTO de puntos a seleccionar. Se usa cuando el tamano
               del coreset no se deriva de |caracteristicas| sino de otra magnitud
               (|C_emb| = 1% de TODOS los parches, |C_dist| = 2048 fijo).
    devuelve : indices (LongTensor) de los m puntos seleccionados.
    '''
    N = caracteristicas.shape[0]
    m = int(n_seleccion) if n_seleccion is not None else max(1, int(round(N * razon)))
    m = max(1, min(m, N))
    red = _proyectar_jl(caracteristicas, dim_proyeccion, dispositivo, semilla)   # (N, d') en device

    generador = torch.Generator(device=dispositivo).manual_seed(semilla)
    inicio = torch.randint(0, N, (1,), generator=generador, device=dispositivo).item()

    seleccionado = torch.empty(m, dtype=torch.long, device=dispositivo)
    seleccionado[0] = inicio
    # distancias^2 de todos al primer punto
    d_min = ((red - red[inicio:inicio + 1]) ** 2).sum(dim=1)

    for i in tqdm(range(1, m), desc="coreset", leave=False):
        indice = torch.argmax(d_min)                 # el mas alejado de lo ya elegido
        seleccionado[i] = indice
        d = ((red - red[indice:indice + 1]) ** 2).sum(dim=1)
        d_min = torch.minimum(d_min, d)           # actualiza cobertura
        d_min[indice] = -1.0                         # evita reelegirlo

    return seleccionado.cpu()


# %% Banco de memoria

class BancoDeMemoria:
    '''Contenedor de los DOS coresets de una categoria (PNI).

    PNI necesita dos niveles de granularidad y no uno:

      C_emb  (self.bank)      banco de embedding. Es contra el que se mide la
                              distancia ||Phi - c||. Grande (~8k vectores).
      C_dist (self.dist_bank) banco de distribucion. Submuestreo de C_emb con
                              el MISMO greedy. Es el "alfabeto" sobre el que el
                              MLP y el histograma estiman probabilidades.
                              Pequeno (~800 vectores) a proposito.

    Por que dos y no uno: el MLP resuelve una clasificacion con |C_dist| clases.
    Con |C_emb| clases el problema seria inabordable y la estimacion de
    probabilidad, puro ruido. Pero reducir C_emb degradaria la precision de la
    distancia. La solucion del paper es desacoplarlos y guardar el mapeo
    emb2dist: a que vector de C_dist corresponde cada vector de C_emb.
    '''

    def __init__(self, conf):
        self.conf = conf
        self.banco = None        # C_emb: (M, C) en CPU
        self.banco_dist = None   # C_dist: (K, C) en CPU
        self.emb2dist = None    # (M,) indice en C_dist de cada vector de C_emb
        self.rejilla = None        # (Hr, Wr) rejilla de parches por imagen
        self.estadisticas = {}         # tiempos y tamanos para el log

    @torch.no_grad()
    def ajustar(self, extractor, cargador_entrenamiento):
        # 1) recolectar los parches de las good
        # Se PREASIGNA el tensor y se rellena por lotes en vez de acumular una
        # lista y hacer torch.cat: el cat necesita la lista Y el resultado vivos
        # a la vez (pico de memoria x2).
        #
        # Dos medidas para que VisA quepa (ver conf.max_parches_banco):
        #   - float16: 1792 dim x 2 bytes = 3,5 KB por parche en vez de 7 KB.
        #     Solo afecta a la SELECCION (comparar distancias); el banco final
        #     se devuelve a float32.
        #   - reservoir sampling: si el total supera el tope, se conserva una
        #     muestra aleatoria UNIFORME de tamano fijo, con semilla propia.
        #     Se usa el algoritmo R de Vitter, que garantiza que cada parche
        #     tenga la misma probabilidad de acabar en el pool sin necesidad de
        #     conocer N de antemano ni de materializarlo entero.
        t0 = time.time()
        tope = int(self.conf.max_parches_banco)
        caracteristicas, desp, n_completo, conservar = None, 0, 0, None
        for imagenes in tqdm(cargador_entrenamiento, desc="extrayendo parches", leave=False):
            parches, rejilla = extractor.extraer_parches(imagenes)
            self.rejilla = rejilla
            parches = parches.cpu().to(torch.float16)

            if caracteristicas is None:
                # Todas las imagenes producen la misma rejilla, asi que el total
                # se conoce exacto tras el primer lote. Con N conocido no hace
                # falta reservoir: basta sortear DE UNA VEZ que indices globales
                # se conservan. El resultado es una muestra uniforme sin
                # reemplazo, y el sorteo es una sola operacion vectorizada.
                n_total = len(cargador_entrenamiento.dataset) * rejilla[0] * rejilla[1]
                n_candidatos = min(n_total, tope)
                if n_total > tope:
                    g = torch.Generator().manual_seed(self.conf.semilla + 7)
                    sel = torch.randperm(n_total, generator=g)[:tope]
                    conservar = torch.zeros(n_total, dtype=torch.bool)
                    conservar[sel] = True
                caracteristicas = torch.empty(n_candidatos, parches.shape[1], dtype=torch.float16)

            n_b = parches.shape[0]
            if conservar is not None:
                parches = parches[conservar[n_completo:n_completo + n_b]]
            caracteristicas[desp:desp + parches.shape[0]] = parches
            desp += parches.shape[0]
            n_completo += n_b
            del parches
        caracteristicas = caracteristicas[:desp]                       # (N_pool, C) en float16
        t_extraccion = time.time() - t0

        # 2) C_emb: submuestreo por greedy coreset
        # m se calcula sobre el numero REAL de parches vistos (n_completo), no sobre
        # el pool: el tope limita entre CUANTOS candidatos se elige, no CUANTOS
        # se eligen. Asi el banco no encoge por una decision de hardware.
        t1 = time.time()
        m = max(1, int(round(n_completo * self.conf.razon_coreset)))
        m = min(m, self.conf.max_coreset, caracteristicas.shape[0])
        indice = coreset_voraz(caracteristicas, None, self.conf.dim_proyeccion,
                             self.conf.dispositivo, self.conf.semilla, n_seleccion=m)
        self.banco = caracteristicas[indice].float().contiguous()   # vuelta a float32
        t_coreset = time.time() - t1

        # 3) C_dist: submuestreo DE C_emb con el mismo criterio
        # Se calcula sobre C_emb (no sobre feats) para que C_dist sea literalmente
        # un subconjunto de C_emb, condicion que exige el mapeo emb2dist.
        # Tamano FIJO |C_dist| = 2048 (seccion 4.1 del paper): es el numero de
        # clases del MLP y no debe variar de una categoria a otra.
        t2 = time.time()
        k_dist = min(int(self.conf.tam_banco_dist), self.banco.shape[0])
        ind_dist = coreset_voraz(self.banco, None, self.conf.dim_proyeccion,
                              self.conf.dispositivo, self.conf.semilla + 1, n_seleccion=k_dist)
        self.banco_dist = self.banco[ind_dist].contiguous()

        # 4) mapeo C_emb -> C_dist
        # Cada vector del banco grande se asigna a su representante mas cercano
        # en el banco pequeno. Se calcula por lotes: la matriz completa
        # (8000 x 800) cabe, pero con coresets mayores no tendria por que.
        self.emb2dist = self._mapear_a_dist(self.banco, self.banco_dist)
        t_dist = time.time() - t2

        self.estadisticas = {
            "n_full": n_completo,
            "n_pool": caracteristicas.shape[0],
            "n_coreset": self.banco.shape[0],
            "n_dist": self.banco_dist.shape[0],
            "dim": self.banco.shape[1],
            "grid": self.rejilla,
            "t_extract_s": round(t_extraccion, 2),
            "t_coreset_s": round(t_coreset, 2),
            "t_dist_s": round(t_dist, 2),
        }
        return self.estadisticas

    @torch.no_grad()
    def _mapear_a_dist(self, emb, dist):
        '''Indice del vector de C_dist mas cercano a cada vector de C_emb.'''
        disp = self.conf.dispositivo
        dist_d = dist.to(disp)
        salida = torch.empty(emb.shape[0], dtype=torch.long)
        tam_bloque = 2048
        for i in range(0, emb.shape[0], tam_bloque):
            bloque = emb[i:i + tam_bloque].to(disp)
            salida[i:i + tam_bloque] = torch.cdist(bloque, dist_d).argmin(dim=1).cpu()
        return salida

    @torch.no_grad()
    def asignar_dist(self, caracteristicas):
        '''Indice en C_dist del vector mas cercano a cada feature dada.

        Se usa para construir las etiquetas del MLP y el histograma posicional:
        cada parche de entrenamiento se etiqueta con el simbolo de C_dist que
        mejor lo representa.
        '''
        return self._mapear_a_dist(caracteristicas, self.banco_dist)

    def guardar(self, ruta):
        torch.save({"bank": self.banco, "dist_bank": self.banco_dist,
                    "emb2dist": self.emb2dist, "grid": self.rejilla,
                    "stats": self.estadisticas}, ruta)

    def cargar(self, ruta):
        d = torch.load(ruta, map_location="cpu")
        self.banco, self.rejilla, self.estadisticas = d["bank"], d["grid"], d["stats"]
        self.banco_dist = d.get("dist_bank")
        self.emb2dist = d.get("emb2dist")
        return self


if __name__ == "__main__":
    # Prueba de humo SIN dataset: parches sinteticos para validar el coreset.
    #   python -m modelo_semisupervisado.banco_memoria
    from modelo_semisupervisado.configuracion import obtener_configuracion, fijar_semilla
    conf = obtener_configuracion()
    fijar_semilla(conf.semilla)

    N, C = 50000, 1792
    falso = torch.randn(N, C)
    t = time.time()
    indice = coreset_voraz(falso, conf.razon_coreset, conf.dim_proyeccion,
                         conf.dispositivo, conf.semilla)
    print(f"banco completo : {N} vectores (dim {C})")
    print(f"coreset ({conf.razon_coreset:.0%}) : {len(indice)} vectores")
    print(f"indices unicos : {len(torch.unique(indice))} (deben ser todos distintos)")
    print(f"tiempo         : {time.time()-t:.2f}s en {conf.dispositivo}")

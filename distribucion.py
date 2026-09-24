'''
Modulo (c-bis): distribucion de normalidad condicionada a posicion y vecindad.

Es el nucleo de PNI y lo que lo separa de un banco de memoria clasico.

    Banco clasico:  "este parche se parece a algo que vi entrenando" -> normal.
    PNI:            "este parche se parece a algo que vi entrenando Y ADEMAS
                     ese algo es plausible AQUI, dado lo que hay alrededor".

Formalmente, en vez de estimar p(Phi(x)) se estima p(Phi(x) | Omega), donde
Omega = (posicion, vecindad). El paper aproxima la probabilidad de cada simbolo
del alfabeto C_dist como la media de dos estimaciones independientes:

    p(c | Omega) ~ [ p(c | N_p(x)) + p(c | x) ] / 2                    (eq. 6)

Este modulo implementa las dos:

  p(c | N_p(x))  -> MLPVecindad. Un MLP que recibe las features VECINAS
                    (ventana p x p sin el centro) y predice que simbolo de
                    C_dist deberia ocupar el centro. Aprende regularidades del
                    tipo "si alrededor hay cable trenzado, en el centro toca
                    cable, no fondo".

  p(c | x)       -> HistogramaPosicional. Para cada posicion de la rejilla, cuenta
                    con que frecuencia aparecio cada simbolo durante el
                    entrenamiento. Captura que un transistor siempre esta en el
                    centro, o que el borde de la pieza cae siempre en el mismo
                    anillo de pixeles.

Ambas se estiman SOLO con imagenes good: no hay fuga de informacion.

--------------------------------------------------------------------------
NOTA SOBRE p (conf.vecindad_dist). El paper lo fija en 9 y ese valor NO es
un detalle menor: es lo que hace que el problema sea de contexto y no trivial.
Las features ya llevan un AvgPool2d de 3x3 encima, asi que con p=3 los vecinos
solapan con el centro en 6 de sus 9 celdas y el MLP puede resolverlo copiando.
Con p=9 la ventana cubre el 15% del ancho de la imagen y el centro queda
realmente PREDICHO por su entorno.

Consecuencia de ingenieria: con p=9 y 1792 canales, la vecindad de UNA posicion
son (81-1)*1792 = 143.360 numeros. Las 3600 posiciones de una sola imagen
ocuparian 2,1 GB si se materializaran a la vez, y F.unfold sobre el lote entero
pediria 8 GB. Por eso las vecindades NO se construyen con unfold sino que se
recogen por bloques de posiciones (recoger_vecindades), tanto al entrenar
como al inferir.
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


# %% Utilidades de vecindad

def rellenar_mapa(mapa, p):
    '''Relleno con ceros de radio p//2, comun a entrenamiento e inferencia.'''
    relleno = p // 2
    return F.pad(mapa, (relleno, relleno, relleno, relleno))


def recoger_vecindades(rellenado, p, ind_b, ind_h, ind_w):
    '''Vecindades p x p SIN el centro, para un CONJUNTO DE POSICIONES dado.

    padded : (B, C, H+2*pad, W+2*pad) mapa YA rellenado con ceros
    ind_b, ind_h, ind_w : (k,) coordenadas en la rejilla SIN padding
    ret    : (k, (p*p-1)*C)

    Se indexa directamente en vez de usar F.unfold porque unfold materializa
    TODAS las ventanas del lote, y con p=9 eso son gigabytes (ver cabecera).
    Aqui el coste es proporcional a k, que elige el llamador.

    El centro se elimina siempre: el MLP debe PREDECIRLO, asi que verlo seria un
    atajo trivial y el modelo no aprenderia nada util.
    '''
    k = ind_b.shape[0]
    desp = torch.arange(p, device=rellenado.device)
    filas = ind_h[:, None] + desp[None, :]                 # (k, p)
    columnas = ind_w[:, None] + desp[None, :]                 # (k, p)
    # Indexacion avanzada separada por un slice: las dimensiones de los indices
    # van primero -> (k, p, p, C).
    vec = rellenado[ind_b[:, None, None], :, filas[:, :, None], columnas[:, None, :]]
    vec = vec.reshape(k, p * p, -1)
    indice_centro = (p * p) // 2
    conservar = torch.arange(p * p, device=rellenado.device) != indice_centro
    return vec[:, conservar, :].reshape(k, -1)


# %% Probabilidad condicionada a la posicion

class HistogramaPosicional:
    '''Histograma de simbolos de C_dist por posicion de la rejilla.

    Implementa el Algoritmo 1 del paper. Para cada imagen de entrenamiento y
    cada posicion x se busca el simbolo de C_dist mas cercano y se incrementa
    su contador. El paper acumula ademas en la VECINDAD p x p de x, no solo en
    x: con pocas imagenes de entrenamiento un histograma por posicion aislada
    seria una estimacion demasiado dispersa.

    Vive en GPU: con |C_dist| = 2048 y rejilla 60x60, el one-hot de un lote son
    118 MB y el promediado local 9x9 son ~600 M de operaciones por imagen. En
    CPU eso costaba segundos POR IMAGEN; en GPU es despreciable.
    '''

    def __init__(self, rejilla, n_simbolos, vecindad=9, dispositivo="cpu"):
        self.H, self.W = rejilla
        self.K = n_simbolos
        self.p = vecindad
        self.dispositivo = dispositivo
        self.hist = torch.zeros(n_simbolos, self.H, self.W, device=dispositivo)

    def acumular(self, ind_simb, tam_lote):
        '''ind_simb: (B*H*W,) simbolo asignado a cada posicion de un LOTE.'''
        B, H, W, K = tam_lote, self.H, self.W, self.K
        simb = ind_simb.to(self.dispositivo).reshape(B, 1, H, W)
        codificacion = torch.zeros(B, K, H, W, device=self.dispositivo)
        codificacion.scatter_(1, simb, 1.0)
        if self.p > 1:
            # Difusion a la vecindad p x p: equivale a sumar el voto de x en
            # todas las posiciones de su ventana. El factor 1/(p*p) es constante
            # dentro de cada posicion, asi que se cancela al normalizar.
            codificacion = F.avg_pool2d(codificacion, kernel_size=self.p, stride=1,
                                  padding=self.p // 2, count_include_pad=False)
        self.hist += codificacion.sum(dim=0)

    def finalizar(self, eps=1e-8):
        '''Normaliza a probabilidad por posicion -> (H*W, K).'''
        probabilidades = self.hist / (self.hist.sum(dim=0, keepdim=True) + eps)
        return probabilidades.permute(1, 2, 0).reshape(self.H * self.W, self.K).contiguous()


# %% Probabilidad condicionada a la vecindad

class MLPVecindad(nn.Module):
    '''MLP que estima p(c_dist | N_p(x)) a partir de las features vecinas.

    Arquitectura del paper: N_MLP capas secuenciales con BatchNorm y ReLU entre
    ellas, y |C_dist| nodos de salida. Se entrena con cross-entropy contra el
    simbolo de C_dist mas proximo a la feature central (etiqueta one-hot).
    '''

    def __init__(self, dim_entrada, n_simbolos, ocultas=1024, n_capas=4):
        super().__init__()
        capas, d = [], dim_entrada
        for _ in range(max(1, n_capas - 1)):
            capas += [nn.Linear(d, ocultas), nn.BatchNorm1d(ocultas),
                       nn.ReLU(inplace=True)]
            d = ocultas
        capas += [nn.Linear(d, n_simbolos)]
        self.red = nn.Sequential(*capas)

    def forward(self, x):
        return self.red(x)


# %% Distribucion condicional combinada

class DistribucionCondicional:
    '''Orquesta el ajuste y la consulta de p(c | Omega).

    fit()     acumula el histograma (una pasada por todas las good) y entrena el
              MLP (varias epocas sobre muestras de posiciones).
    predict() devuelve p(c_dist | Omega) para cada posicion de una imagen.
    '''

    def __init__(self, conf, banco_memoria):
        self.conf = conf
        self.bm = banco_memoria
        self.rejilla = banco_memoria.rejilla
        self.K = banco_memoria.banco_dist.shape[0]
        self.mlp = None
        self.probs_pos = None       # (H*W, K)

    # Ajuste
    def ajustar(self, extractor, cargador_entrenamiento, registro=None):
        conf = self.conf

        # 1) histograma posicional: una pasada por TODAS las good
        if conf.usar_posicion:
            histograma = HistogramaPosicional(self.rejilla, self.K, conf.vecindad_dist,
                                     conf.dispositivo)
            for imagenes in tqdm(cargador_entrenamiento, desc="histograma (P)", leave=False):
                mapa = extractor(imagenes)
                B, C, _, _ = mapa.shape
                plano = mapa.permute(0, 2, 3, 1).reshape(-1, C)
                histograma.acumular(self.bm.asignar_dist(plano.cpu()), B)
                del mapa, plano
            self.probs_pos = histograma.finalizar()
            del histograma

        # 2) MLP de vecindad
        if conf.usar_vecindad:
            self._entrenar_mlp(extractor, cargador_entrenamiento, registro)
        return self

    def _entrenar_mlp(self, extractor, cargador_entrenamiento, registro=None):
        '''Entrena p(c|N_p(x)) generando las vecindades al vuelo.

        PRESUPUESTO FIJO DE MUESTRAS. Materializar el conjunto de entrenamiento
        es imposible (cada muestra son 143.360 floats). Se genera al vuelo, pero
        ademas se acota a conf.muestras_por_epoca_mlp muestras por epoca y se
        sortea QUE IMAGENES se leen en cada epoca. Dos motivos:

          - Comparabilidad: una categoria de VisA tiene 900 imagenes de
            entrenamiento y una de MVTec ~200. Sin el tope, el MLP de VisA veria
            4,7 veces mas datos por epoca y las dos mitades del estudio dejarian
            de compartir protocolo.
          - Coste: el cuello de botella es pasar el backbone por las imagenes en
            CADA epoca. Leyendo solo las que hacen falta para llenar el cupo, el
            coste por epoca deja de crecer con el tamano del dataset.

        Como cada epoca sortea imagenes y posiciones distintas, a lo largo del
        entrenamiento se cubre todo el conjunto.
        '''
        conf = self.conf
        disp = conf.dispositivo
        p = conf.vecindad_dist
        H, W = self.rejilla
        dim_entrada = (p * p - 1) * self.bm.banco.shape[1]

        self.mlp = MLPVecindad(dim_entrada, self.K, conf.ocultas_mlp,
                                   conf.capas_mlp).to(disp)
        optimizador = torch.optim.Adam(self.mlp.parameters(), lr=conf.tasa_mlp)
        planificador = torch.optim.lr_scheduler.StepLR(optimizador, step_size=conf.paso_tasa_mlp,
                                                gamma=conf.gamma_tasa_mlp)

        conj = cargador_entrenamiento.dataset
        pos_por_imagen = max(1, int(H * W * conf.submuestreo_pos_mlp))
        n_imagenes = min(len(conj),
                     max(1, math.ceil(conf.muestras_por_epoca_mlp / pos_por_imagen)))

        for ep in range(conf.epocas_mlp):
            g = torch.Generator().manual_seed(conf.semilla * 1000 + ep)
            imagenes_elegidas = torch.randperm(len(conj), generator=g)[:n_imagenes].tolist()
            cargador = DataLoader(Subset(conj, imagenes_elegidas), batch_size=conf.tam_lote,
                                shuffle=False, num_workers=0)

            self.mlp.train()
            tot, vistos = 0.0, 0
            for imagenes in tqdm(cargador, desc=f"MLP {ep+1}/{conf.epocas_mlp}",
                             leave=False):
                with torch.no_grad():
                    mapa = extractor(imagenes)                  # (B, C, H, W) en dev
                    B, C, _, _ = mapa.shape
                    plano = mapa.permute(0, 2, 3, 1).reshape(-1, C)
                    y_todos = self.bm.asignar_dist(plano.cpu()).to(disp)  # (B*H*W,)
                    rellenado = rellenar_mapa(mapa, p)
                    del mapa, plano

                # posiciones sorteadas de este lote
                n = B * H * W
                k = min(n, pos_por_imagen * B)
                sel = torch.randperm(n, generator=g)[:k].to(disp)
                ind_b = torch.div(sel, H * W, rounding_mode="floor")
                resto = sel % (H * W)
                ind_h = torch.div(resto, W, rounding_mode="floor")
                ind_w = resto % W

                for i in range(0, k, conf.lote_mlp):
                    sl = slice(i, i + conf.lote_mlp)
                    if ind_b[sl].numel() < 2:    # BatchNorm necesita >=2
                        continue
                    with torch.no_grad():
                        X = recoger_vecindades(rellenado, p, ind_b[sl],
                                                 ind_h[sl], ind_w[sl])
                    yb = y_todos[sel[sl]]
                    perdida = F.cross_entropy(self.mlp(X), yb)
                    optimizador.zero_grad(set_to_none=True)
                    perdida.backward()
                    optimizador.step()
                    tot += perdida.item() * yb.numel()
                    vistos += yb.numel()
                    del X
                del rellenado, y_todos
            planificador.step()
            if registro:
                registro.info(f"      MLP epoca {ep+1}/{conf.epocas_mlp} "
                            f"loss={tot/max(vistos,1):.4f} ({vistos} muestras, "
                            f"{n_imagenes} img, lr={planificador.get_last_lr()[0]:.1e})")
        self.mlp.eval()

    # Consulta
    @torch.no_grad()
    def predecir(self, mapa):
        '''p(c_dist | Omega) para cada posicion de UNA imagen -> (H*W, K).'''
        conf = self.conf
        H, W = self.rejilla
        disp = conf.dispositivo
        partes = []

        if conf.usar_vecindad and self.mlp is not None:
            p = conf.vecindad_dist
            rellenado = rellenar_mapa(mapa.to(disp), p)
            hh = torch.arange(H, device=disp).repeat_interleave(W)
            ww = torch.arange(W, device=disp).repeat(H)
            bb = torch.zeros(H * W, dtype=torch.long, device=disp)
            probabilidades = torch.empty(H * W, self.K, device=disp)
            # Por bloques: 3600 vecindades de 143.360 floats serian 2,1 GB.
            for i in range(0, H * W, conf.trozo_vecindad):
                sl = slice(i, i + conf.trozo_vecindad)
                X = recoger_vecindades(rellenado, p, bb[sl], hh[sl], ww[sl])
                # Temperature scaling: divide los logits antes del softmax. Sin
                # el, la red satura en 0/1 y el umbral tau no filtraria nada.
                probabilidades[sl] = F.softmax(self.mlp(X) / conf.temperatura, dim=1)
                del X
            partes.append(probabilidades)
            del rellenado

        if conf.usar_posicion and self.probs_pos is not None:
            partes.append(self.probs_pos.to(disp))

        if not partes:                        # ablation sin N ni P: uniforme
            return torch.full((H * W, self.K), 1.0 / self.K, device=disp)
        if len(partes) == 1:
            return partes[0]
        return torch.stack(partes, dim=0).mean(dim=0)     # eq. 6

    # Guardado
    def state_dict(self):
        return {"mlp": None if self.mlp is None else self.mlp.state_dict(),
                "pos_probs": self.probs_pos, "K": self.K, "grid": self.rejilla}


if __name__ == "__main__":
    # Prueba de humo SIN dataset ni GPU:
    #   .\venv\Scripts\python.exe -m modelo_semisupervisado.distribucion
    from modelo_semisupervisado.configuracion import obtener_configuracion, fijar_semilla
    conf = obtener_configuracion()
    conf.dispositivo = "cpu"
    fijar_semilla(conf.semilla)

    B, C, H, W, K, p = 2, 16, 8, 8, 12, 9
    mapa = torch.randn(B, C, H, W)
    rellenado = rellenar_mapa(mapa, p)
    b = torch.tensor([0, 1, 1])
    h = torch.tensor([0, 3, 7])
    w = torch.tensor([0, 4, 7])
    vec = recoger_vecindades(rellenado, p, b, h, w)
    print(f"mapa            : {tuple(mapa.shape)}  padded {tuple(rellenado.shape)}")
    print(f"vecindades      : {tuple(vec.shape)}  (esperado 3 x {(p*p-1)*C})")
    assert vec.shape == (3, (p * p - 1) * C)

    # el centro NO debe aparecer en su propia vecindad
    centro = mapa[1, :, 3, 4]
    trozos = vec[1].reshape(p * p - 1, C)
    assert not any(torch.allclose(trozos[i], centro) for i in range(p * p - 1)), \
        "el centro no puede formar parte de su propia vecindad"
    print("centro excluido : OK")

    # validacion cruzada del indexado contra F.unfold (con p pequeno)
    p2 = 3
    relleno2 = rellenar_mapa(mapa, p2)
    ref = F.unfold(mapa, kernel_size=p2, padding=p2 // 2).view(B, C, p2 * p2, H * W)
    ref = ref[:, :, [i for i in range(p2 * p2) if i != 4], :]
    ref = ref.permute(0, 3, 2, 1).reshape(B, H * W, -1)
    obtenido = recoger_vecindades(relleno2, p2, b, h, w)
    exp = torch.stack([ref[b[i], h[i] * W + w[i]] for i in range(3)])
    assert torch.allclose(obtenido, exp), "gather debe coincidir con unfold"
    print("vs F.unfold     : identico  [OK]")

    histograma = HistogramaPosicional((H, W), K, p, "cpu")
    histograma.acumular(torch.randint(0, K, (B * H * W,)), B)
    probabilidades = histograma.finalizar()
    print(f"histograma      : {tuple(probabilidades.shape)}  "
          f"suma por posicion={probabilidades.sum(dim=1).mean():.4f} (debe ser 1.0)")
    assert torch.allclose(probabilidades.sum(dim=1), torch.ones(H * W), atol=1e-5)

    mlp = MLPVecindad((p * p - 1) * C, K, ocultas=32, n_capas=4)
    print(f"salida MLP      : {tuple(mlp(vec).shape)}  (esperado 3 x {K})")
    print("OK")

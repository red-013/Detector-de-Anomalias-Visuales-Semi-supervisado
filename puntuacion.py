'''
Modulo (d): inferencia y scoring de anomalias segun PNI.

CAMBIO DE FONDO respecto al banco de memoria clasico. Antes el score de un
parche era su distancia al vecino mas cercano del banco:

    S(x) = min_{c in C_emb} ||Phi(x) - c||

Ahora el score es la log-verosimilitud negativa de la feature CONDICIONADA a la
posicion y la vecindad (ecuaciones 2, 4, 5 y 8 del paper):

    p(Phi(x) | c)      ~ exp(-lambda * ||Phi(x) - c||)              (eq. 8)
    p(Phi(x) | Omega)  ~ max_{c in C_emb} p(Phi(x)|c) * T_tau(p(c|Omega))   (eq. 4)
    S(x)               = -log p(Phi(x) | Omega)                     (eq. 2)

con la funcion umbral T_tau(v) = 1 si v > tau, y 0 en otro caso (eq. 5).

Sustituyendo, y como -log(exp(-d)) = d, el score colapsa a algo muy legible:

    S(x) = min de ||Phi(x) - c||  RESTRINGIDO a los c cuya probabilidad
           condicional p(c|Omega) supera tau

Es decir: la misma distancia de siempre, pero los vectores del banco que son
IMPLAUSIBLES en esa posicion y vecindad quedan fuera de la competicion. Un
parche defectuoso que casualmente se parece a un trozo de banco de otra region
de la imagen ya no puede refugiarse en el: ese candidato esta vetado ahi.

Ese veto es exactamente lo que aporta el +0.52 de I-AUROC de la ablation.

tau = factor_tau / |C_emb|. El paper usa tau = 1/(2|C_emb|), valor deliberado:
cualquier tau por debajo de 1/|C_emb| garantiza que al menos un candidato
sobreviva al filtro y el minimo nunca quede indefinido. Aun asi se implementa un
respaldo explicito por si una fila queda vacia.
'''
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


# %% Calculo del score de anomalia

class Evaluador:
    def __init__(self, banco_memoria, conf, distribucion=None, banco_neg=None):
        self.conf = conf
        self.banco = banco_memoria.banco.to(conf.dispositivo)        # C_emb: (M, C)
        self.rejilla = banco_memoria.rejilla                       # (Hr, Wr)
        self.dist = distribucion                           # p(c|Omega) o None
        self.emb2dist = (banco_memoria.emb2dist.to(conf.dispositivo)
                         if banco_memoria.emb2dist is not None else None)
        # tau: umbral de plausibilidad, escalado al tamano del banco.
        self.tau = conf.factor_tau / self.banco.shape[0]
        # C_neg: banco de parches de defecto etiquetados (regimen semisupervisado).
        # None -> el score es el de PNI sin modificar.
        self.banco_neg = (banco_neg.to(conf.dispositivo)
                         if (banco_neg is not None and conf.beta_neg > 0) else None)

    @torch.no_grad()
    def _puntajes_parche(self, parches, mapa):
        '''Distancia al candidato plausible mas cercano, por parche.

        patches : (P, C) features de la imagen, aplanadas
        fmap    : (1, C, H, W) el mismo mapa sin aplanar (lo pide el MLP)
        ret     : (P,) distancias
        '''
        D = torch.cdist(parches, self.banco)                # (P, M)

        # Sin modelo de distribucion -> comportamiento clasico (baseline de la
        # ablation: la fila "sin N ni P" de la Tabla 3 del paper).
        if self.dist is None or self.emb2dist is None:
            return self._contrastar(D.min(dim=1).values, parches)

        # p(c_dist | Omega) por posicion -> (P, K)
        probs_dist = self.dist.predecir(mapa)               # (P, K)

        # Se traslada la probabilidad del simbolo a cada vector de C_emb via el
        # mapeo emb2dist, por bloques de filas: la matriz (P, M) completa a
        # 60x60 y 8k vectores ocupa ~115 MB, y con bancos mayores no cabria.
        P = parches.shape[0]
        salida = torch.empty(P, device=parches.device)
        lam = self.conf.lambda_exp
        eps = 1e-30
        tam_bloque = 512
        for i in range(0, P, tam_bloque):
            pe = probs_dist[i:i + tam_bloque][:, self.emb2dist]    # (b, M)
            d = D[i:i + tam_bloque]

            # T_tau (eq. 5) se usa SOLO como poda: el paper la presenta como una
            # forma de "reducir el tiempo de computo con una pequena perdida de
            # rendimiento", no como el mecanismo que aporta la mejora. Anular los
            # candidatos implausibles equivale a asignarles p(c|Omega) = 0.
            if self.conf.modo_puntaje == "max":
                # Eq. 4 literal del paper: maximo con umbral binario. El score
                # queda en la distancia al candidato PLAUSIBLE mas cercano.
                #   S = -log max_c [ exp(-lambda*d_c) * T_tau(p_c) ]
                #     = min { d_c : p_c > tau }
                S = d.masked_fill(pe <= self.tau, float("inf")).min(dim=1).values
            else:
                # Eq. 3: suma ponderada por la probabilidad condicional.
                #   S = -log sum_c exp(-lambda*d_c) * p_c
                #     = -logsumexp(-lambda*d_c + log p_c)
                # A diferencia del maximo, aqui un candidato cercano pero poco
                # probable contribuye poco y el score sube de forma continua.
                log_p = torch.log(pe.clamp_min(eps))
                log_p = log_p.masked_fill(pe <= self.tau, float("-inf"))
                S = -torch.logsumexp(-lam * d + log_p, dim=1)

            # Respaldo: si una fila quedo sin ningun candidato plausible, se cae
            # al minimo libre (comportamiento del banco clasico).
            malos = ~torch.isfinite(S)
            if malos.any():
                S[malos] = d[malos].min(dim=1).values
            salida[i:i + tam_bloque] = S
        return self._contrastar(salida, parches)

    @torch.no_grad()
    def _contrastar(self, s_pos, parches):
        '''Termino contrastivo del regimen semisupervisado: s = d+ - beta * d-.

        Interpretacion. `s_pos` mide cuanto se aleja el parche de la normalidad;
        `d_neg` mide cuanto se aleja del defecto conocido mas proximo. Restar el
        segundo penaliza a los parches que, ademas de parecerse poco a lo
        normal, se parecen MUCHO a un defecto ya visto, y deja practicamente
        intactos a los que estan lejos de ambas nubes.

        El resultado se acota a >= 0. Sin esa cota, un parche normal situado
        casualmente cerca de C_neg obtendria puntuacion negativa y quedaria por
        debajo del resto de parches normales, invirtiendo el orden dentro de la
        clase conforme sin ninguna justificacion.

        Se aplica sobre el score YA calculado por PNI, no dentro de la eq. 4:
        asi el filtro de plausibilidad T_tau sigue operando exactamente como en
        el articulo y el efecto de las etiquetas queda separado del suyo.
        '''
        if self.banco_neg is None:
            return s_pos
        d_neg = torch.empty_like(s_pos)
        bloque = 4096
        for i in range(0, parches.shape[0], bloque):
            d_neg[i:i + bloque] = torch.cdist(
                parches[i:i + bloque], self.banco_neg).min(dim=1).values
        return (s_pos - self.conf.beta_neg * d_neg).clamp_min(0.0)

    @torch.no_grad()
    def puntuar(self, extractor, imagen):
        '''
        img : (1, 3, H, W)
        devuelve:
          image_score : float             -> AUROC imagen
          mapa_pixeles   : (H, W) np.ndarray -> AUROC pixel / heatmap
        '''
        mapa = extractor(imagen)                              # (1, C, Hr, Wr)
        _, C, Hr, Wr = mapa.shape
        parches = mapa.permute(0, 2, 3, 1).reshape(-1, C).to(self.conf.dispositivo)

        d_parche = self._puntajes_parche(parches, mapa)        # (P,)

        # Score de imagen: el maximo simple. Se mantiene la decision tomada en
        # el sweep de 2026-08-22 (la reponderacion de PatchCore perjudicaba,
        # p=0.0026); PNI tampoco la usa, su score de imagen es el maximo del
        # mapa de anomalia.
        puntaje_imagen = d_parche.max().item()

        # Mapa de pixeles: reshape -> interpolar -> gaussiano
        baja_res = d_parche.reshape(1, 1, Hr, Wr)
        ampliado = F.interpolate(baja_res, size=(self.conf.tam_recorte, self.conf.tam_recorte),
                           mode="bilinear", align_corners=False)
        ampliado = ampliado.squeeze().cpu().numpy()
        mapa_pixeles = gaussian_filter(ampliado, sigma=self.conf.sigma_gaussiano)
        return puntaje_imagen, mapa_pixeles

    @torch.no_grad()
    def puntuar_cargador(self, extractor, cargador_prueba):
        '''Recorre el cargador_prueba y devuelve listas alineadas para el modulo (e).'''
        puntajes_imagen, etiquetas, mapas_pixeles, mascaras = [], [], [], []
        for imagen, etiqueta, mascara in cargador_prueba:
            s, mapa_p = self.puntuar(extractor, imagen)
            puntajes_imagen.append(s)
            etiquetas.append(int(etiqueta.item()))
            mapas_pixeles.append(mapa_p)
            mascaras.append(mascara.squeeze().numpy())
        return puntajes_imagen, etiquetas, mapas_pixeles, mascaras


if __name__ == "__main__":
    # Prueba de humo SIN dataset: verifica que el filtro por plausibilidad
    # hace lo que promete.
    #   .\venv\Scripts\python.exe -m modelo_semisupervisado.puntuacion
    from types import SimpleNamespace
    from modelo_semisupervisado.configuracion import obtener_configuracion, fijar_semilla
    conf = obtener_configuracion()
    conf.dispositivo = "cpu"
    conf.tam_recorte = 64
    fijar_semilla(conf.semilla)

    C, M, K, H, W = 32, 200, 20, 8, 8
    banco = torch.randn(M, C)
    bm = SimpleNamespace(banco=banco, rejilla=(H, W),
                         emb2dist=torch.randint(0, K, (M,)))

    # 1) sin distribucion -> debe reducirse al vecino mas cercano clasico
    evaluador = Evaluador(bm, conf, distribucion=None)
    mapa = torch.randn(1, C, H, W)
    parches = mapa.permute(0, 2, 3, 1).reshape(-1, C)
    d_base = evaluador._puntajes_parche(parches, mapa)
    d_ref = torch.cdist(parches, banco).min(dim=1).values
    assert torch.allclose(d_base, d_ref), "sin distribucion debe ser el NN clasico"
    print(f"baseline (NN clasico)   : media={d_base.mean():.4f}")

    # 2) distribucion UNIFORME: no aporta informacion. En modo "max" el orden
    #    de los parches debe conservarse exactamente (el AUROC solo depende del
    #    orden); en modo "sum" el score es un promedio suave y no tiene por que.
    class DistribucionUniforme:
        def predecir(self, _mapa):
            return torch.full((H * W, K), 1.0 / K)
    conf.modo_puntaje = "max"
    d_uni = Evaluador(bm, conf, distribucion=DistribucionUniforme())._puntajes_parche(parches, mapa)
    assert torch.allclose(d_uni, d_base),         "modo max con p uniforme debe reducirse al NN clasico"
    print(f"uniforme (modo max)     : identico al baseline  [OK]")

    # 3) veto SELECTIVO: se anula el simbolo del vecino mas cercano en la mitad
    #    de las posiciones. Esas posiciones deben subir su score frente a las
    #    no vetadas, que es justo el efecto que persigue PNI.
    vecino = torch.cdist(parches, banco).argmin(dim=1)
    simb_vecino = bm.emb2dist[vecino]
    marcadas = torch.zeros(H * W, dtype=torch.bool)
    marcadas[::2] = True

    class DistribucionVeto:
        def predecir(self, _mapa):
            p = torch.full((H * W, K), 1.0 / K)
            p[marcadas, simb_vecino[marcadas]] = 0.0
            return p

    for modo in ("max", "sum"):
        conf.modo_puntaje = modo
        base = Evaluador(bm, conf, distribucion=DistribucionUniforme())._puntajes_parche(parches, mapa)
        veto = Evaluador(bm, conf, distribucion=DistribucionVeto())._puntajes_parche(parches, mapa)
        sv = (veto - base)[marcadas].mean().item()
        sl = (veto - base)[~marcadas].mean().item()
        print(f"modo {modo:<4}: subida vetadas={sv:+.4f}  libres={sl:+.4f}")
        assert sv > sl, f"modo {modo}: vetar el NN debe penalizar esa posicion"
    print("OK")

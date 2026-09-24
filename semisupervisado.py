'''
Modulo (c-ter): componentes del regimen semisupervisado (OE3, resultado R5).

El pipeline base es PNI (Bae et al., ICCV 2023): un banco de memoria de
caracteristicas preentrenadas mas el modelado de la normalidad condicionada a
la posicion y la vecindad. PNI opera con muestras conformes unicamente. Este
modulo anade lo que el OE3 exige y PNI no contempla: el uso de un subconjunto
reducido de muestras ANOMALAS ETIQUETADAS, el definido en el resultado R4.

Tres componentes, cada uno con un peso propio que puede anularse:

  (1) BANCO NEGATIVO C_neg
      Los parches de defecto reales se almacenan en un segundo banco, analogo
      a C_emb pero de la clase contraria. Es la estructura nueva; las otras dos
      la explotan.

  (2) DEPURACION SUPERVISADA DE C_emb
      SoftPatch (Wang et al.) observa que el banco de memoria supone un
      entrenamiento libre de defectos y propone ponderar los parches para
      descartar los que no representan normalidad, estimando el ruido SIN
      supervision mediante deteccion de atipicos. Aqui el problema es el
      simetrico y la senal es directa: con defectos etiquetados se puede
      identificar que parches del banco positivo estan tan cerca de un defecto
      conocido que lo TAPAN. Mientras esos parches sigan en C_emb, un defecto
      de ese tipo encuentra siempre un vecino cercano y su distancia no sube.

  (3) SCORE CONTRASTIVO
      La anomalia deja de medirse solo como distancia a lo normal y pasa a
      contrastar esa distancia con la distancia al defecto conocido mas
      proximo: s = d+ - beta * d-, acotado a >= 0.

Ninguno de los tres altera la formulacion de PNI cuando su peso es cero, de
modo que el regimen one-class se recupera exactamente y las dos mitades del
estudio comparten protocolo, conformes y conjunto de prueba.
'''
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from modelo_semisupervisado.banco_memoria import coreset_voraz


# %% Banco negativo de parches de defecto

@torch.no_grad()
def extraer_parches_defecto(extractor, cargador_anomalas, conf, registro=None):
    '''Parches de DEFECTO de las imagenes anomalas etiquetadas.

    Una imagen anomala es mayoritariamente normal: si se tomasen todos sus
    parches, la mayor parte de C_neg seria fondo sano y el banco negativo
    dejaria de discriminar -peor aun, el termino contrastivo restaria
    puntuacion a parches perfectamente normales-. Por eso se conserva unicamente
    el parche cuya celda de la rejilla esta cubierta por la mascara al menos en
    `conf.cobertura_mascara_neg`.

    La mascara se reduce de la resolucion de entrada a la de la rejilla con un
    promediado por area (`adaptive_avg_pool2d`), de forma que el valor de cada
    celda ES la fraccion de pixeles anomalos que contiene. Interpolar con
    NEAREST daria un binario que ignora el tamano del defecto dentro de la celda.

    ret: (N, C) float16 en CPU, o None si ninguna celda supera el umbral.
    '''
    caracteristicas, n_imagenes, n_celdas = [], 0, 0
    for imagenes, etiquetas, mascaras in tqdm(cargador_anomalas, desc="parches de defecto",
                                    leave=False):
        fusionado = extractor(imagenes)                          # (B, C, Hr, Wr)
        B, C, Hr, Wr = fusionado.shape
        m = mascaras.to(fusionado.device).float()
        if m.dim() == 3:
            m = m.unsqueeze(1)                           # (B, 1, H, W)
        cobertura = F.adaptive_avg_pool2d(m, (Hr, Wr))         # fraccion por celda
        sel = (cobertura.reshape(B, -1) >= conf.cobertura_mascara_neg)   # (B, Hr*Wr)

        parches = fusionado.permute(0, 2, 3, 1).reshape(B, Hr * Wr, C)
        for b in range(B):
            if sel[b].any():
                caracteristicas.append(parches[b][sel[b]].cpu().to(torch.float16))
                n_celdas += int(sel[b].sum())
        n_imagenes += B
        del fusionado, parches

    if not caracteristicas:
        if registro:
            registro.warning("  [semi] ninguna celda supera la cobertura %.2f: "
                           "los defectos son menores que una celda de la rejilla",
                           conf.cobertura_mascara_neg)
        return None
    salida = torch.cat(caracteristicas, dim=0)
    if registro:
        registro.info("  [semi] %d parches de defecto de %d imagenes (%.1f por imagen)",
                    salida.shape[0], n_imagenes, salida.shape[0] / max(1, n_imagenes))
    return salida


@torch.no_grad()
def construir_banco_negativo(parches_defecto, conf, registro=None):
    '''C_neg por greedy coreset, con el mismo criterio que C_emb.

    Se usa el mismo algoritmo que para el banco positivo -y no un muestreo
    aleatorio- para que ambos bancos cubran su respectivo espacio con el mismo
    criterio de dispersion: si uno se construyese por k-center y el otro al
    azar, la distancia a cada uno no seria comparable y el termino contrastivo
    quedaria sesgado por la forma de haberlos construido.
    '''
    if parches_defecto is None:
        return None
    k = min(int(conf.max_banco_neg), parches_defecto.shape[0])
    if k < parches_defecto.shape[0]:
        indice = coreset_voraz(parches_defecto, None, conf.dim_proyeccion,
                             conf.dispositivo, conf.semilla + 3, n_seleccion=k)
        neg = parches_defecto[indice]
    else:
        neg = parches_defecto
    neg = neg.float().contiguous()
    if registro:
        registro.info("  [semi] |C_neg| = %d (de %d candidatos)",
                    neg.shape[0], parches_defecto.shape[0])
    return neg


# %% Purga del banco positivo

@torch.no_grad()
def purgar_banco_positivo(banco, banco_neg, conf, registro=None):
    '''Elimina de C_emb los parches que tapan defectos conocidos.

    Para cada vector p de C_emb se comparan dos distancias:
        d_neg  = distancia al defecto conocido mas cercano
        d_pos  = distancia a su vecino mas cercano DENTRO de C_emb, excluyendose
                 a si mismo (mide la densidad local de la nube normal)
    Se marca p si  d_neg < margen_purga * d_pos, esto es, si p esta mas cerca de
    un defecto de lo que lo esta de su propio vecindario normal.

    Dos salvaguardas, ambas necesarias:
      - Se purga como maximo `razon_purga` de C_emb, y se eliminan los casos mas
        extremos primero (menor cociente d_neg/d_pos). Sin el tope, una
        categoria cuyos defectos se parezcan mucho a la normalidad podria perder
        una fraccion grande del banco y degradar la deteccion en vez de
        mejorarla.
      - Nunca se vacia el banco: se conserva siempre el resto.

    ret: (bank_depurado, n_purgados)
    '''
    if banco_neg is None or conf.razon_purga <= 0:
        return banco, 0
    disp = conf.dispositivo
    B = banco.to(disp)
    N = banco_neg.to(disp)

    d_neg = torch.cdist(B, N).min(dim=1).values           # (M,)
    # Vecino mas cercano dentro del propio banco, sin contarse a si mismo: la
    # diagonal de la matriz de distancias es cero por construccion.
    dd = torch.cdist(B, B)
    dd.fill_diagonal_(float("inf"))
    d_pos = dd.min(dim=1).values
    del dd

    razon = d_neg / d_pos.clamp_min(1e-12)
    cand = razon < conf.margen_purga
    n_max = int(conf.razon_purga * B.shape[0])
    n_purgar = int(min(cand.sum().item(), n_max))
    if n_purgar == 0:
        if registro:
            registro.info("  [semi] purga: 0 parches (ningun candidato)")
        return banco, 0

    orden = torch.argsort(razon)                          # los mas extremos primero
    quitar = orden[:n_purgar]
    conservar = torch.ones(B.shape[0], dtype=torch.bool, device=disp)
    conservar[quitar] = False
    salida = banco[conservar.cpu()].contiguous()
    if registro:
        registro.info("  [semi] purga: %d de %d parches (%.2f %%), %d candidatos",
                    n_purgar, B.shape[0], 100.0 * n_purgar / B.shape[0],
                    int(cand.sum().item()))
    return salida, n_purgar


@torch.no_grad()
def distancia_negativa(parches, banco_neg, dispositivo, trozo=4096):
    '''Distancia de cada parche al defecto conocido mas cercano.

    Se calcula por bloques porque en inferencia `parches` son las 3600
    posiciones de una imagen y |C_neg| puede llegar a 4096: la matriz completa
    son 59 MB por imagen, asumible, pero el troceado mantiene el coste acotado
    si cualquiera de los dos crece.
    '''
    if banco_neg is None:
        return None
    N = banco_neg.to(dispositivo)
    salida = torch.empty(parches.shape[0], device=dispositivo)
    for i in range(0, parches.shape[0], trozo):
        salida[i:i + trozo] = torch.cdist(parches[i:i + trozo], N).min(dim=1).values
    return salida


# %% Umbral balanceado

def umbral_balanceado(puntajes_validacion, puntajes_anomalas):
    '''Umbral de decision calibrado con las dos clases DE ENTRENAMIENTO.

    El pipeline one-class solo podia calibrar con conformes (percentil de las
    apartadas de validacion), y eso hacia que la tasa de falsa alarma fijada a
    priori no se sostuviera sobre el conjunto de prueba: sin ejemplos de la
    clase positiva no hay forma de saber donde cae realmente la frontera.

    Aqui se dispone de anomalas etiquetadas que NO estan en el test, de modo que
    el punto de operacion puede elegirse maximizando el F1 sobre ellas sin
    ninguna fuga de informacion. La diferencia con el best-F1 habitual de la
    literatura es exactamente esa: aquel se calcula sobre el conjunto de prueba.

    ret: (umbral, f1_en_calibracion)
    '''
    import numpy as np
    v = np.asarray(puntajes_validacion, dtype=float)
    a = np.asarray(puntajes_anomalas, dtype=float)
    if v.size == 0 or a.size == 0:
        return float(np.percentile(v, 90.0)) if v.size else 0.0, float("nan")

    cand = np.unique(np.concatenate([v, a]))
    mejor_f1, mejor_t = -1.0, float(cand[0])
    for t in cand:
        tp = float((a > t).sum())
        fp = float((v > t).sum())
        fn = float((a <= t).sum())
        f1 = 2 * tp / max(1e-12, 2 * tp + fp + fn)
        if f1 > mejor_f1:
            mejor_f1, mejor_t = f1, float(t)
    return mejor_t, mejor_f1


# %% Orquestacion del regimen semisupervisado

@torch.no_grad()
def ajustar_semisupervisado(extractor, cargador_anomalas, banco_memoria, conf, registro=None):
    '''Orquesta los tres componentes. Devuelve un dict con C_neg y estadisticas.

    Se ejecuta DESPUES de BancoDeMemoria.fit y ANTES de entrenar la distribucion
    condicionada, porque la purga cambia C_emb y el mapeo emb2dist debe
    recalcularse sobre el banco ya depurado.
    '''
    estadisticos = {"n_neg": 0, "n_purged": 0, "t_semi_s": 0.0}
    if conf.regimen != "semi" or cargador_anomalas is None:
        return None, estadisticos

    t0 = time.time()
    defecto = extraer_parches_defecto(extractor, cargador_anomalas, conf, registro)
    neg = construir_banco_negativo(defecto, conf, registro)
    del defecto

    if neg is not None:
        banco_memoria.banco, n_purgados = purgar_banco_positivo(
            banco_memoria.banco, neg, conf, registro)
        estadisticos["n_purged"] = n_purgados
        if n_purgados:
            # C_dist es un SUBCONJUNTO de C_emb y emb2dist mapea uno en otro:
            # si se purga C_emb sin rehacer el mapeo, las probabilidades del
            # MLP se asignarian a vectores que ya no estan.
            banco_memoria.emb2dist = banco_memoria._mapear_a_dist(
                banco_memoria.banco, banco_memoria.banco_dist)
        estadisticos["n_neg"] = int(neg.shape[0])
    estadisticos["t_semi_s"] = round(time.time() - t0, 2)
    return neg, estadisticos

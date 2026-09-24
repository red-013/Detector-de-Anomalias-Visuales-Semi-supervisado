'''
Modulo (b): extractor de caracteristicas jerarquicas multiescala.

Es el nucleo del OE2. Tres sub-decisiones, todas implementadas aqui:

  1) QUE CAPAS.   Extraemos layer1 + layer2 + layer3 de una WideResNet-50
     preentrenada en ImageNet (>= 3 niveles jerarquicos -> OE2). Descartamos:
       - layer4: demasiado semantica/sesgada a clases ImageNet y 7x7 (mala
         localizacion).
       - (solo layer3, como haria un metodo de una sola capa): pierde el
         detalle fino que necesitamos para la AUROC de pixel.
     El PatchCore original usa 2 capas (layer2+layer3); aqui lo AMPLIAMOS a 3
     incorporando layer1 (bajo nivel). Es un aporte propio y medible (ablation
     2 vs 3 niveles).

  2) AGREGACION LOCAL.   Cada vector de parche se hace "consciente de su
     vecindad" promediando una ventana p x p (AvgPool2d stride=1, padding=p//2).
     No cambia la resolucion; aumenta robustez y campo receptivo efectivo.

  3) FUSION MULTIESCALA.   Los mapas tienen resoluciones distintas (56/28/14).
     Se interpolan bilinealmente a la resolucion de 'capa_fusion' (por defecto
     layer2 = 28x28) y se concatenan por canales. Resultado: un descriptor por
     posicion que combina las 3 escalas.

     Por que fusionar a 28x28 y no a 56x56 (la de layer1):
       - 28x28 mantiene el tamano del banco IGUAL al PatchCore canonico
         (Colab / GPU modesta friendly) y ya alcanza ~98% AUROC pixel.
       - 56x56 cuadruplica el numero de parches (mas RAM/tiempo) sin ganancia
         garantizada. Si quieres probarlo, cambia conf.capa_fusion="layer1".
     Es decir: seguimos EXTRAYENDO 3 niveles (OE2), pero controlamos el coste
     eligiendo la resolucion de fusion.

La red va CONGELADA y en eval(): no hay backprop en ninguna etapa (requisito
one-class y de coste computacional: solo inferencia).
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# %% Extractor jerarquico multiescala

class ExtractorJerarquico(nn.Module):
    def __init__(self, nombre_backbone="wide_resnet50_2",
                 capas=("layer1", "layer2", "layer3"),
                 capa_fusion="layer2", vecindad=3, dispositivo="cpu"):
        super().__init__()
        self.capas = tuple(capas)
        self.capa_fusion = capa_fusion
        self.dispositivo = dispositivo
        assert capa_fusion in self.capas, \
            f"fusion_layer '{capa_fusion}' debe estar en layers {self.capas}"

        # Backbone preentrenado en ImageNet, congelado.
        backbone = getattr(torchvision.models, nombre_backbone)(
            weights="IMAGENET1K_V1")
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone.to(dispositivo)

        # Hooks: capturan la salida de cada etapa sin tocar el forward de la red.
        self._caracteristicas = {}
        for nombre in self.capas:
            getattr(self.backbone, nombre).register_forward_hook(self._gancho(nombre))

        # Agregacion de vecindario local (no cambia H,W).
        self.agrupamiento = nn.AvgPool2d(kernel_size=vecindad, stride=1,
                                 padding=vecindad // 2)

    def _gancho(self, nombre):
        def fn(_modulo, _entrada, salida):
            self._caracteristicas[nombre] = salida          # (B, C, H, W)
        return fn

    @torch.no_grad()
    def forward(self, x):
        '''Devuelve el mapa fusionado (B, C_total, Hr, Wr).'''
        x = x.to(self.dispositivo)
        self._caracteristicas.clear()
        self.backbone(x)                        # dispara los hooks

        # 2) agregacion local en cada escala, a su resolucion nativa
        caracteristicas = [self.agrupamiento(self._caracteristicas[n]) for n in self.capas]

        # 3) alinear a la resolucion de fusion e interpolar bilinealmente
        tam_ref = self._caracteristicas[self.capa_fusion].shape[-2:]
        caracteristicas = [F.interpolate(f, size=tam_ref, mode="bilinear",
                               align_corners=False) for f in caracteristicas]
        return torch.cat(caracteristicas, dim=1)          # concat por canales

    @torch.no_grad()
    def extraer_parches(self, x):
        '''
        Aplana el mapa fusionado a una matriz de parches.
        Devuelve:
          patches : (B*Hr*Wr, C_total)  -> lo que se guarda en el banco
          grid    : (Hr, Wr)            -> para reconstruir el mapa de anomalia
        '''
        fusionado = self.forward(x)                 # (B, C, Hr, Wr)
        B, C, H, W = fusionado.shape
        parches = fusionado.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
        return parches, (H, W)

    @torch.no_grad()
    def dim_caracteristicas(self):
        '''Dimension C_total del descriptor de parche (util para el proyector JL).'''
        ficticio = torch.zeros(1, 3, 224, 224, device=self.dispositivo)
        return self.forward(ficticio).shape[1]


if __name__ == "__main__":
    # Prueba de humo (no requiere el dataset; usa una imagen aleatoria):
    #   python -m modelo_semisupervisado.extractor_caracteristicas
    from modelo_semisupervisado.configuracion import obtener_configuracion, fijar_semilla
    conf = obtener_configuracion()
    fijar_semilla(conf.semilla)

    extractor = ExtractorJerarquico(
        conf.backbone, conf.capas, conf.capa_fusion,
        conf.vecindad, dispositivo=conf.dispositivo)

    x = torch.randn(2, 3, conf.tam_recorte, conf.tam_recorte)
    fusionado = extractor(x)
    parches, rejilla = extractor.extraer_parches(x)
    print(f"backbone      : {conf.backbone}  capas={conf.capas}")
    print(f"mapa fusionado: {tuple(fusionado.shape)}  (fusion en {conf.capa_fusion})")
    print(f"grid          : {rejilla}  -> {rejilla[0]*rejilla[1]} parches/imagen")
    print(f"parches (lote): {tuple(parches.shape)}  dim descriptor={extractor.dim_caracteristicas()}")

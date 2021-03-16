import torch
import torch.nn as nn

from distillation.module import DistillerModule, LayerMap, LayerConfig


class BaseDistiller(nn.Module):
    def __init__(self, teacher, student, layer_map):
        self.distiller = DistillerModule(teacher, student)
        self.layer_map = layer_map
        
        super(BaseDistiller, self).__init__()




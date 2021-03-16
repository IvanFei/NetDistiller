import torch
import traceback
import torch.nn as nn
from termcolor import colored
from collections import defaultdict, namedtuple

from distillation.hook import Hook, DistillerHookContext


class LayerConfig(object):

    r""" Layer Config contain info: student_layer, teacher_layer, loss_fn, alpha.

        Now only support init and get item, forbid to change the value after init.

        Args:
            cfg (list): list contained for value, e.g. [student_layer, teacher_layer, loss_fn, alpha]

        Example::
            >>> layer_config = LayerConfig([1, 1, nn.CrossEntropyLoss(), 0.5])
            >>> layer_config
            LayerConfig(teacher_layer=1, student_layer=1, loss_fn=CrossEntropyLoss(), alpha=0.5)

    """

    _key_idx_map = {
        "teacher_layer": 0,
        "student_layer": 1,
        "loss_fn": 2,
        "alpha": 3
    }

    def __init__(self, cfg):

        self._Layer = namedtuple("Layer", list(LayerConfig._key_idx_map.keys()))

        self._check_cfg(cfg)
        self._container = self._Layer(*cfg)

    @staticmethod
    def _check_cfg(cfg):
        assert isinstance(cfg, (tuple, list)), "Init the LayerConfig should be list"
        assert len(cfg) == 4, \
            "The Len of cfg should contain 4 args, e.g. `[student_layer, teacher_layer, loss_fn, alpha]`"

    def __getitem__(self, item):

        if isinstance(item, int):
            return self._container[item]
        elif isinstance(item, str):
            assert item in LayerConfig._key_idx_map.keys(), \
                "Item should be one of `[student_layer, teacher_layer, loss_fn, alpha]`"
            return self._container[LayerConfig._key_idx_map[item]]
        else:
            raise KeyError("`item` should be int or string.")

    def __repr__(self):
        return "LayerConfig(teacher_layer={}, " \
               "student_layer={}, " \
               "loss_fn={}, " \
               "alpha={})".format(self._container[0], self._container[1],
                                  self._container[2], self._container[3])


class LayerMap(object):
    r""" Config of the student and teacher layer

        This would be useful for whole training step.

        Args:
            teacher_layers (list or int): teacher layers id
            student_layers (list or int): student layers id
            layer_losses (list or function): loss function, default `nn.CrossEntropyLoss`
            alphas (list or float): loss weight, default `1`

        Example::
            >>> layer_map = LayerMap([1, 2, 3], [1, 2, 3])
            >>> layer_map.append([4, 4, nn.BCELoss(), 0.5])
            >>> layer_map[0]
            LayerConfig(teacher_layer=1, student_layer=1, loss_fn=CrossEntropyLoss(), alpha=1)
    """
    def __init__(self,
                 teacher_layers,
                 student_layers,
                 layer_losses=nn.CrossEntropyLoss(),
                 alphas=1):
        assert type(teacher_layers) == type(student_layers), \
            "Type of `student_layer` should the same as `teacher_layer`."
        if isinstance(teacher_layers, int):
            teacher_layers, student_layers = [teacher_layers, ], [student_layers, ]

        assert len(teacher_layers) == len(student_layers), \
            "Len of `teacher_layer` should equal with `student_layer`"

        if not isinstance(layer_losses, (list, tuple)):
            layer_losses = [layer_losses for _ in range(len(teacher_layers))]

        if isinstance(alphas, (int, float)):
            alphas = [alphas for _ in range(len(teacher_layers))]

        self.config = list()
        for cfg in zip(teacher_layers, student_layers, layer_losses, alphas):
            self.config.append(
                LayerConfig(list(cfg))
            )

    def get(self, index):

        return self.__getitem__(index)

    def append(self, cfg):
        if isinstance(cfg, list):
            cfg = LayerConfig(cfg)

        assert isinstance(cfg, LayerConfig), "cfg should be list or LayerConfig."

        self.config.append(cfg)

    def __getitem__(self, index):
        return self.config[index]

    def __len__(self):
        return len(self.config)

    def __delitem__(self, key):
        del self.config[key]

    def __repr__(self):
        rep = "LayerMap: "
        for idx, cfg in enumerate(self.config):
            rep += "\n({}): {}".format(idx, cfg)
        return rep


class DistillerModule(nn.Module):
    r""" Distiller Module for knowledge distillation.

        Involve studentNet and teacherNet.

        Args:
            student (nn.Module): student net
            teacher (nn.Module): teacher net

        Returns:
            preds (dict): forward return dict with keys
                `["teacher_pred", "student_pred", "teacher_feat", "student_feat"]`
    """
    def __init__(self, teacher, student):
        super(DistillerModule, self).__init__()
        self.student = student
        self.teacher = teacher
        self.student_modules = [*self.student.children()]
        self.teacher_modules = [*self.teacher.children()]
        self.layer_map = None
        self.context = None
        self.student_hooks = None
        self.teacher_hooks = None

    def forward(self, *inputs, **kwargs):
        t_out = self.teacher(*inputs, **kwargs)
        s_out = self.student(*inputs, **kwargs)

        t_feat, s_feat = self.context.values()

        return {
            "teacher_pred": t_out,
            "student_pred": s_out,
            "teacher_feat": t_feat,
            "student_feat": s_feat
        }

    def register_layers(self, layer_map):
        assert isinstance(layer_map, LayerMap), "`layer_map` should be class LayerMap."
        self.layer_map = layer_map
        self.context = DistillerHookContext(self.layer_map)
        self.context.set_hooks(self.teacher, self.student)

    @staticmethod
    def _repr(teacher_modules, student_modules):
        len_teacher, len_student = len(teacher_modules), len(student_modules)
        rep = "DistillerModule(\n"

        for idx in range(max(len_teacher, len_student)):
            rep += "({}): \n\t".format(idx)
            if idx < len_teacher:
                rep += "Teacher({})".format(teacher_modules[idx]) \
                    if isinstance(teacher_modules[idx], nn.Sequential) \
                    else "Teacher({})".format(str(teacher_modules[idx].replace("\n", "")))
            else:
                rep += "Teacher()"

            if idx < len_student:
                rep += "Student()".format(student_modules[idx]) \
                    if isinstance(student_modules[idx], nn.Sequential) \
                    else "Teacher({})".format(str(student_modules[idx].replace("\n", "")))
            else:
                rep += "Student()"

            rep += "\n"

        rep += ")"
        return rep

    def __repr__(self):
        return self._repr(self.teacher_modules, self.student_modules)



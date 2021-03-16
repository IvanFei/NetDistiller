

# ref: https://github.com/Kennethborup/knowledgeDistillation
class Hook(object):
    r""" A simple hook class that return the forward results.

        Now not support the backward hook.
    """
    def __init__(self):
        self.output = None

    def hook_fn(self, module, input, output):
        """ hook function to save the output"""
        self.output = output

    def set_hook(self, module):
        """ Attaches hook to model """
        self.hook = module.register_forward_hook(self.hook_fn)

    @property
    def val(self):
        """ Return the hook output"""
        return self.output

    @property
    def hooked(self):
        """ Return True if hook function is set."""
        return hasattr(self, "hook")


class DistillerHookContext(object):

    r""" Distiller Hook Context.

        To hook the mid features in network.

        Args:
            layer_map (LayerMap): list contain LayerConfig.

        Example::
            >>> from distillation.module import LayerMap
            >>> from torchvision.models.resnet import resnet34, resnet50
            >>> teacher, student = resnet50(), resnet34()
            >>> layer_map = LayerMap([1, 2], [1, 2])
            >>> context = DistillerHookContext(layer_map=layer_map)
            >>> context.set_hooks(student, teacher)
            >>> teachers, students = context.values
    """

    def __init__(self, layer_map):
        self.layer_map = layer_map

        self.teacher_layers = [layer["teacher_layer"] for layer in self.layer_map]
        self.student_layers = [layer["student_layer"] for layer in self.layer_map]

        self.teacher_hooks = [Hook() for _ in self.layer_map]
        self.student_hooks = [Hook() for _ in self.layer_map]

    def set_hooks(self, teacher, student):
        if not self.teacher_hooks[0].hooked:
            for hook, layer in zip(self.teacher_hooks, self.teacher_layers):
                hook.set_hook(DistillerHookContext._get_module(teacher, layer))

        if not self.student_hooks[0].hooked:
            for hook, layer in zip(self.student_hooks, self.student_layers):
                hook.set_hook(DistillerHookContext._get_module(student, layer))

    @property
    def values(self):
        teacher_vals = [hook.val for hook in self.teacher_hooks]
        student_vals = [hook.val for hook in self.student_hooks]
        return teacher_vals, student_vals

    @staticmethod
    def _get_module(network, layer):
        assert isinstance(layer, int), "Layer in LayerMap should be `int`."
        children = [*network.children()]
        return children[layer]


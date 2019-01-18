import math
import operator
from collections import defaultdict
from functools import reduce
from typing import Type, Union

from torch.nn import Module
import torch


def profile_modules(enable=True, profile_gpu=True, skip_first=True) -> Type[Module]:
    """
    decorator for nn.Module. This method decorate the nn.Module
    adding layer by layer profile capability. Based on torch.autograd.profiler.

    :param skip_first: Skip the first time measurement (Cuda takes a long time to init)
    :param profile_gpu: Boolean flag profile the gpu
    :param enable: Boolean flag profile the gpu
    :return: nn.Module with profiling capabilities
    """

    def wrap(module_class: Type[Module]):
        """
        :param module_class: nn.Module
        :param module_class:
        :return:
        """
        assert issubclass(module_class, Module), "profile can only wrap torch.nn.Module class"
        module_original_init = module_class.__init__

        def wrap_init_(self, *args, **kwargs):
            module_original_init(self, *args, **kwargs)
            self.profiler = ModuleProfiler(self, enable=enable, profile_gpu=profile_gpu, skip_first=skip_first)
            if enable:
                module_class.__str__ = self.profiler.__str__

        module_class.__init__ = wrap_init_
        return module_class

    return wrap


class ModuleProfiler:
    def __init__(self, module: Module, enable=True, profile_gpu=True, skip_first=True):
        """
        The profiling done only on the module basic operations, i.e. modules that were defined
        in the init and do not have children (Functions and other non module operation in forward pass
        will not be measured). Modules that have children are counted as some of all of their children

        .. warning: (requirement by torch.autograd.profiler.profile)
        This context manager should not be called recursively, i.e. at most one
        instance should be enabled at any given time.

        :param module: main module
        :param profile_gpu: profile gpu operations
        :param skip_first: skip the first call
        :param enable: enable profiling
        """
        self._skip_first = skip_first
        self._module = module
        self._enable = enable
        self._profile_gpu = profile_gpu
        self._forward_event = defaultdict(ProfilerEvent)
        self._called_first = defaultdict(lambda: True)
        self._container_hits = defaultdict(lambda: 0)
        self._operations = {}
        self._containers = {}

        # Apply forward hook for the module operations and it's children
        list(map(self.hook_operation, ModuleProfiler.operations(self._module)))

        # Apply forward hook for containers
        list(map(self.hook_containers, self._module.modules()))

    def hook_containers(self, operation: Module):
        def wrapper_forward(op: Module, *input, **kwargs):
            """
            wrapper_forward will wrap the forward method of container to count number of calls
            :param op: module
            :param input: module inputs
            :param kwargs: module kwargs inputs
            :return:
            """
            if self._skip_first and self._called_first[op]:
                self._called_first[op] = False
            else:
                self._container_hits[op] += 1
            return self._containers[op.__class__](op, *input, **kwargs)

        if len(list(operation.children())):
            # wrap __call__ of nn.Module. store the original __Call__ operation
            if operation.__class__ not in self._containers:
                self._containers[operation.__class__] = operation.__class__.__call__
                operation.__class__.__call__ = wrapper_forward

    def hook_operation(self, operation):
        def wrapper_forward(op: Module, *input, **kwargs):
            """
            wrapper_forward will wrap the forward method with autograd.profiler.profile to count time.
            :param op: module
            :param input: module inputs
            :param kwargs: module kwargs inputs
            :return:
            """

            if not self._enable:  # profiler is not enabled
                return self._operations[op.__class__](op, *input, **kwargs)

            with torch.autograd.profiler.profile(use_cuda=self._profile_gpu) as prof:
                result = self._operations[op.__class__](op, *input, **kwargs)

            # update the ForwardEvent
            if self._skip_first and self._called_first[op]:
                self._called_first[op] = False
            else:
                self._forward_event[op] += ProfilerEvent(cpu_time=prof.total_average().cpu_time,
                                                         gpu_time=prof.total_average().cuda_time,
                                                         parameters=count_elements(op.parameters()),
                                                         input_size=count_elements(input),
                                                         flops=ModuleProfiler.flops(op, result),
                                                         hits=1)
            return result

        # wrap __call__ of nn.Module. store the original __Call__ operation
        if operation.__class__ not in self._operations:
            self._operations[operation.__class__] = operation.__class__.__call__
            operation.__class__.__call__ = wrapper_forward

    @staticmethod
    def flops(module: Module, output: torch.Tensor):
        """
        calculate the number of operations
        :param output: output tensor
        :param module: input module
        :return: number of flops
        """
        if isinstance(module, torch.nn.Conv2d):
            kernel_height, kernel_width = module.kernel_size
            batch_sz, _, height, width = output.size()
            in_channels = module.in_channels
            out_channels = module.out_channels

            flops_per_location = batch_sz * kernel_height * kernel_width * in_channels * out_channels
            if module.bias is not None:
                flops_per_location += batch_sz * out_channels

            return flops_per_location * height * width
        elif isinstance(module, torch.nn.Linear):
            batch_sz = output.size(0)
            in_channels = module.in_features
            out_channels = module.out_features

            flops = batch_sz * in_channels * out_channels
            if module.bias is not None:
                flops += out_channels

            return flops
        # todo Add support for additional module flop calculations
        elif isinstance(module, torch.nn.BatchNorm2d):
            pass
        elif isinstance(module, torch.nn.MaxPool2d):
            pass
        elif isinstance(module, (torch.nn.ReLU, torch.nn.PReLU, torch.nn.ELU, torch.nn.LeakyReLU, torch.nn.ReLU6)):
            pass
        return 0  # None supported flops

    @staticmethod
    def operations(module: Module):
        """
        Given a module recursively transverse it
        to find all atomic operations.

        Atomic operations are the nodes in the graph which
        perform computations on the tensors (i.e. they are nn.Module which are not containers, sequential, etc.)
        :param module: module for which a recursive operations are searched
        """
        if not len(list(module.children())):
            # nn.Module who doesn't have sub nn.Module, hook it.
            yield module

        for name, sub_module in module.named_children():
            if (isinstance(sub_module, torch.nn.Container)
                    or isinstance(sub_module, torch.nn.Sequential)
                    or isinstance(sub_module, torch.nn.ModuleList)
                    or isinstance(sub_module, torch.nn.Module)):
                # Recursively visit their descendants.
                yield from ModuleProfiler.operations(sub_module)

    def get_metrics(self, module):
        if module in self._forward_event:
            # it's an operation
            return self._forward_event[module]

        # it's a type of container
        container_forward_event = reduce(ProfilerEvent.simple_add, map(self.get_metrics, module.children()))
        container_forward_event.hits = self._container_hits[module]
        return container_forward_event

    def __str__(self, module=None, indentation=0, pre_msg=''):
        tmpstr = ''
        if module is None:
            module = self._module
            tmpstr += ProfilerEvent.header()

        # this is an operation
        metrics = self.get_metrics(module).tostring()

        if module.__class__ in self._operations:
            return tmpstr + metrics + indent(pre_msg + module.__repr__(), indentation) + '\n'

        name = module.__class__.__name__
        tmpstr += metrics + indent(pre_msg + name + '(', indentation) + '\n'
        for key, sub_module in module._modules.items():
            tmpstr += self.__str__(sub_module, indentation + 2, pre_msg='(' + key + '): ')
        tmpstr += indent(')', indentation + len(metrics)) + '\n'
        return tmpstr


class ProfilerEvent:
    """
    ProfilerEvent logs a profiling event
    """

    def __init__(self, cpu_time=0, gpu_time=0, parameters=0, flops=0, input_size=0, hits=0):
        self.flops = flops
        self.cpu_time = cpu_time
        self.gpu_time = gpu_time
        self.parameters = parameters
        self.input_size = input_size
        self.hits = hits

    @staticmethod
    def header():
        header = format_columns(
            ['Avg CPU Time', 'Avg GPU Time', 'hits', 'Total Time', 'Parameters', 'Input', 'FLOPS', 'Architecture'])
        return '\n'.join([header, '=' * len(header), ''])

    def tostring(self):
        return format_columns([
            format_time(self.cpu_time),
            format_time(self.gpu_time),
            str(self.hits),
            format_time((self.cpu_time + self.gpu_time) * self.hits),
            format_count(self.parameters),
            format_count(self.input_size),
            format_flops(self.flops)])

    @staticmethod
    def simple_add(first, second):
        return ProfilerEvent(
            first.cpu_time + second.cpu_time,
            first.gpu_time + second.gpu_time,
            first.parameters + second.parameters,  # parameters are shared. No need to add
            first.flops + second.flops,
            first.input_size + second.input_size)  # input is shared. No need to add

    def __add__(self, other):
        total_cpu_time = (self.cpu_time * self.hits + other.cpu_time * other.hits)
        total_gpu_time = (self.gpu_time * self.hits + other.gpu_time * other.hits)
        return ProfilerEvent(
            total_cpu_time / (self.hits + other.hits),
            total_gpu_time / (self.hits + other.hits),
            other.parameters,  # parameters are shared. No need to add
            other.flops,
            other.input_size,  # input is shared. No need to add
            self.hits + other.hits)

    def __radd__(self, other):
        return self.__add__(other)


def format_columns(cols, width=13):
    assert isinstance(cols, list)
    return ' ' + ' '.join(col.center(width, ' ') for col in cols) + ' '


def format_time(time_in_ns: float):
    if not time_in_ns:
        return '-'

    human_powers = ['n', 'u', 'm', '']
    power = int(math.log(time_in_ns, 10) // 3)
    return '{:.2f}{}s '.format(
        time_in_ns / 1000. ** power,
        human_powers[power])


def format_count(n):
    if not n:
        return '-'

    human_powers = ['', 'KB', 'MB', 'GB']
    power = int(math.log(n, 10) // 3)
    return '{:.2f}{} '.format(
        n / 1000. ** power,
        human_powers[power])


def format_flops(f):
    if not f:
        return '-'

    human_flops = ['', 'Kmac', 'Mmac', 'Gmac']
    power = int(math.log(f, 10) // 3)
    return '{:.2f}{} '.format(
        f / 1000. ** power,
        human_flops[power])


def indent(s, indentation):
    return '\n'.join((indentation * ' ') + line for line in s.split('\n'))


def count_elements(tensors: Union[list, tuple]):
    """
    count tensor elements
    :param tensors:
    :return:
    """

    def bytes_num(tensor: torch.Tensor):
        if tensor.dtype == torch.float32:
            return 4
        elif tensor.dtype == torch.float64:
            return 8
        elif tensor.dtype == torch.float16:
            return 2
        elif tensor.dtype == torch.int8:
            return 1
        else:
            raise Exception('Unknown tensor size')

    # filter inputs that are not tensors
    tensors = filter(lambda x: isinstance(x, torch.Tensor), tensors)
    return sum([bytes_num(t) * reduce(operator.mul, t.size()) for t in tensors])


if __name__ == '__main__':
    from torchvision.models.resnet import ResNet, BasicBlock

    # wrap resnet18 with profiler
    ResNet_profiling = profile_modules(enable=True, skip_first=True)(ResNet)

    # init resnet18
    network = ResNet_profiling(BasicBlock, [2, 2, 2, 2])

    # move network to gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network.to(device)

    input = torch.ones(1, 3, 224, 224, dtype=torch.float, device=device)

    for i in range(10):
        network(input)

    print(network)

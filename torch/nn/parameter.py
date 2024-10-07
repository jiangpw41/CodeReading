"""
原文件223行：5个类，1个函数
class:
    _ParameterMeta(torch._C._TensorMeta)
    Parameter(torch.Tensor, metaclass=_ParameterMeta)
    UninitializedTensorMixin
    UninitializedParameter(UninitializedTensorMixin, Parameter)
    UninitializedBuffer(UninitializedTensorMixin, torch.Tensor)
function:
    is_lazy(param)

"""



import torch
from torch._C import _disabled_torch_function_impl
from collections import OrderedDict

# 元类Metaclass to combine _TensorMeta and the instance check override for Parameter.
class _ParameterMeta(torch._C._TensorMeta):
    """
    torch._C._TensorMeta是PyTorch 用于张量类型的元类。在父类功能的基础上，让 isinstance(obj, Parameter) 判断更为灵活，即使对象 obj 是自定义的 Tensor 类型，
    但只要它有 _is_param 标志，也会被认为是 Parameter 类型。这对于自定义张量类型尤其有用，因为它们可能需要与 Parameter 一样的行为。
    """
    def __instancecheck__(self, instance):
        return super().__instancecheck__(instance) or (
            isinstance(instance, torch.Tensor) and getattr(instance, '_is_param', False))

# 继承自
class Parameter(torch.Tensor, metaclass=_ParameterMeta):
    r"""
    作用：当一个 torch.Tensor 被封装在 torch.nn.Parameter 中并赋值为模型的一个属性时，
    这个张量会被自动注册到模型的参数列表中，被用来作为神经网络模型中的可训练参数，
    优化器（如 SGD、Adam 等）在更新模型时会自动找到这些参数（出现在 Module.parameters() 的迭代器中）
    这样一来，模型的可训练参数和hidden_states张量就区分开了。
    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. Note that
            the torch.no_grad() context does NOT affect the default behavior of
            Parameter creation--the Parameter will still have `requires_grad=True` in
            :class:`~no_grad` mode. See :ref:`locally-disable-grad-doc` for more
            details. Default: `True`默认为True需要梯度，
            在上下文管理器with no_grad中也不会变为False，只有显式表示为False
    """

    def __new__(cls, data=None, requires_grad=True):
        # 如果没有提供 data，则使用一个空张量初始化。
        if data is None:
            data = torch.empty(0)
        if type(data) is torch.Tensor or type(data) is Parameter:
            # _make_subclass保持了与标准张量的兼容性。是一个底层的 PyTorch 方法，用于创建一个 Parameter 对象，同时继承 Tensor 的属性
            #　为了保持向后兼容性，继续沿用这种路径。未来可能会统一标准张量的行为。
            return torch.Tensor._make_subclass(cls, data, requires_grad)

        # 从data中分离独立的新副本，下面检查detach后生成的张量的语义
        t = data.detach().requires_grad_(requires_grad)
        if type(t) is not type(data):  
            raise RuntimeError(f"Creating a Parameter from an instance of type {type(data).__name__} "
                               "requires that detach() returns an instance of the same type, but return "
                               f"type {type(t).__name__} was found instead. To use the type as a "
                               "Parameter, please correct the detach() semantics defined by "
                               "its __torch_dispatch__() implementation.")
        t._is_param = True                      # 为什么要强行改成True？
        return t

    # 下面的三个方法只对标准Tensor有效，自定义的Tensor依旧被设定为自定义tensor类型和放啊不会被调用
    def __deepcopy__(self, memo):       # memo 是一个字典，用于存储已经复制过的对象，以避免循环引用和冗余的拷贝操作。
        if id(self) in memo:            # memo中如果已经存在一份拷贝，直接返回
            return memo[id(self)]
        else:                           # 创建 data 的深拷贝，并传入 self.requires_grad 以保持原有的属性。memory_format=torch.preserve_format：确保张量的内存格式被保留preserve。
            result = type(self)(self.data.clone(memory_format=torch.preserve_format), self.requires_grad)
            memo[id(self)] = result     # 将新创建的对象存储到 memo 字典中，供后续引用。
            return result
    
    def __repr__(self):                 # 返回对象的“官方”字符串表
        return 'Parameter containing:\n' + super().__repr__()

    def __reduce_ex__(self, proto):     # 用于对象序列化的一个特殊方法，通过它可以控制对象的序列化和反序列化过程。proto 是序列化协议版本。
        state = torch._utils._get_obj_state(self)       # 获取当前对象的状态信息。

        # 创建一个空的有序字典用于存储钩子（hooks），防止钩子在序列化时被包含。
        hooks = OrderedDict()                           
        if not state:                   # 如果没有对象状态（状态为空），返回不带状态的重建函数和参数。
            return (
                torch._utils._rebuild_parameter,            # 用于反序列化时重建参数的函数。
                (self.data, self.requires_grad, hooks)      # 传递数据、是否需要梯度、钩子作为参数。
            )

        return (                        # 否则返回带状态的重建函数和参数：
            torch._utils._rebuild_parameter_with_state,
            (self.data, self.requires_grad, hooks, state)
        )
    # __torch_function__ 是 PyTorch 中的一个机制，允许用户重载 torch 操作。
    # 将 __torch_function__ 设置为禁用实现。这意味着对于该类的实例，不会应用 __torch_function__ 重载机制。通常用于优化性能或避免在子类中不必要的功能。
    __torch_function__ = _disabled_torch_function_impl

# 这个Mixin提供了对未初始化张量的特殊处理，比如不允许访问形状等属性，以及提供了materialize方法用于将未初始化的张量转换为实际的张量。
class UninitializedTensorMixin:
    '''
    整体来看，这个类是为了在某些特定的上下文中处理未初始化的张量，例如在使用延迟加载模块（LazyModules）时。
    它确保了在张量被实际使用之前，必须先通过forward方法进行初始化。
    这种设计可能是为了优化内存使用，或者在某些情况下延迟参数的初始化，直到实际需要它们。
    '''
    _allowed_methods = [                    # 列出了允许在未初始化的张量上调用的方法。这包括一些基本的张量属性访问和操作，比如size, copy_, cuda, cpu等。
        torch.Tensor.__hash__,
        torch.Tensor.size,
        torch.Tensor.copy_,
        torch.Tensor.is_complex,
        torch.Tensor.is_floating_point,
        torch.Tensor.half,
        torch.Tensor.float,
        torch.Tensor.double,
        torch.Tensor.char,
        torch.Tensor.short,
        torch.Tensor.int,
        torch.Tensor.long,
        torch.Tensor.cuda,
        torch.Tensor.cpu,
        torch.Tensor.to,
        torch.Tensor.get_device,
        torch._has_compatible_shallow_copy_type,
    ]

    def materialize(self, shape, device=None, dtype=None):
        r"""将未初始化的张量“实体化”，即创建一个具有相同属性（设备和数据类型）的初始化张量。
        它接受一个现有张量或者指定的shape, device, 和dtype作为参数，实体化一个一样的张量。
        Args:
            shape : (tuple): the shape for the materialized tensor.
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module. Optional.
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module. Optional.
        """
        if device is None:              # 或未指定设备、类型，与data的设备、类型一致
            device = self.data.device
        if dtype is None:           
            dtype = self.data.dtype     # 创建一个相同的空张量并替换data，替换了之前未初始化的张量数据。
        self.data = torch.empty(shape, device=device, dtype=dtype)
        self.__class__ = self.cls_to_become     # 表示这个未初始化的张量在被实体化后应该转换成的类
        '''
        这种模式允许一个对象在初始化之前保持一种轻量级的状态，直到实际需要它的数据时才进行转换。
        这样做的好处是可以在运行时节省内存，并且可以延迟初始化过程，直到真正需要使用这些数据的时候。
        这在某些深度学习框架中，特别是在处理非常大的模型或者在模型并行化的场景中，是一种常见的优化技术。
        '''

    @property
    def shape(self):
        # 这是一个只读属性，如果尝试访问未初始化张量的形状，会抛出一个RuntimeError。
        raise RuntimeError(
            'Can\'t access the shape of an uninitialized parameter or buffer. '
            'This error usually happens in `load_state_dict` when trying to load '
            'an uninitialized parameter into an initialized one. '
            'Call `forward` to initialize the parameters before accessing their attributes.')

    def share_memory_(self):
        # 如果尝试在未初始化的张量上调用share_memory_，会抛出一个RuntimeError。
        raise RuntimeError(
            'Can\'t share memory on an uninitialized parameter or buffer. '
            'Call `forward` to initialize the parameters before calling '
            '`module.share_memory()`.')

    def __repr__(self):
        "定义了类的字符串表示形式，当打印这个类的实例时会返回<UninitializedTensorMixin>。"
        return f'<{self.__class__.__name__}>'

    def __reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]这个方法是Python的pickle模块用来序列化对象的。这里它返回了一个元组，包含了类本身和需要序列化的数据。
        return (
            self.__class__,
            (self.requires_grad,)
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # method-wrapper is to detect access to Tensor properties that are
        # wrapped in descriptors
        if func in cls._allowed_methods or func.__class__.__name__ == 'method-wrapper':
            if kwargs is None:
                kwargs = {}
            return super().__torch_function__(func, types, args, kwargs)
        raise ValueError(
            f'Attempted to use an uninitialized parameter in {func}. '
            'This error happens when you are using a `LazyModule` or '
            f'explicitly manipulating `torch.nn.parameter.{cls.__name__}` '
            'objects. When using LazyModules Call `forward` with a dummy batch '
            'to initialize the parameters before calling torch functions')

# 判断是否是延迟初始化的LazyModule的标准是，如果是UninitializedTensorMixin的一个实例就是
def is_lazy(param):
    return isinstance(param, UninitializedTensorMixin)


class UninitializedParameter(UninitializedTensorMixin, Parameter):
    r"""表示一个尚未初始化的参数parameter
    继承自torch.nn.Parameter，这意味着它具有PyTorch参数的所有特性，比如自动求导等；
    也有UninitializedTensorMixin指定的一些基础特性，如不允许访问形状等属性
    原注释翻译如下：
    UninitializedParameter是torch.nn.Parameter的特例。其中数据data的形状shape仍然未知。
    与torch.nn.Parameter不同，UninitializedParameter不包含任何数据，试图访问某些属性，如它们的形状，将抛出运行时错误。
    可以对UninitializedParameter执行的唯一操作是更改其数据类型、将其移动到其他设备并将其转换为常规的：torch.nn.Parameter。
    参数实体化materialized时使用的默认设备或数据类型可以在构造过程中使用例如“device='cuda'”进行设置。
    """
    cls_to_become = Parameter       # 指定了当未初始化的参数被实体化（materialize）时，应该转换成的类。UninitializedTensorMixin重写UninitializedTensorMixin中的属性

    def __new__(cls, requires_grad=True, device=None, dtype=None) -> None:
        """
        特殊的静态方法，用于创建类的实例。这个方法接受requires_grad, device, 和 dtype作为参数，
        并使用torch.empty(0, **factory_kwargs)来创建一个形状为0的空张量，
        这个张量将作为UninitializedParameter实例的初始数据。
        torch.Tensor._make_subclass是一个工厂模式，接口用于创建对象，但是让子类决定实例化哪一个类。工厂方法让类的实例化推迟到子类中进行。
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        data = torch.empty(0, **factory_kwargs)
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __deepcopy__(self, memo):
        """
        重写Parameter父类的方法，区别在于它没有显式复制张量的数据和内存格式，而是使用了原始张量的设备和数据类型信息，创建了新对象。
        使用clone()方法的复制可能会占用更多的内存，因为它创建了数据的完整副本，但可以提供更好的性能，特别是在需要保持内存格式一致性的情况下。
        不使用clone()的复制可能在内存使用上更高效，因为它避免了数据的完整复制，但可能在性能上不如前者，特别是在需要频繁访问张量数据的情况下。
        """
        if id(self) in memo:
            return memo[id(self)]
        else:
            #  这行代码使用了Python的反射特性来动态创建对象，先获取了当前对象实例（self）的类，然后使用当前类的构造函数new。
            result = type(self)(self.requires_grad, self.data.device, self.data.dtype)
            memo[id(self)] = result
            return result

class UninitializedBuffer(UninitializedTensorMixin, torch.Tensor):
    r"""
    未初始化的缓冲区buffer。和UninitializedParameter类似，只不过前者是对nn.Parameter的加成，本类是对nn.Tensor的加成
    未初始化缓冲区是：torch.Tensor的一个特例，其中数据的形状仍然未知。

    与class:`torch.Tensor`不同，未初始化的参数不包含任何数据，试图访问某些属性，如它们的形状，将抛出运行时错误。
    可以对未初始化的参数执行的唯一操作是更改其数据类型，将其移动到其他设备并将其转换为常规的class:`torch.Tensor`.张量。
    
    缓冲区实体化时使用的默认设备或数据类型可以在构造过程中使用例如“device='cuda'”进行设置。
    """

    cls_to_become = torch.Tensor

    def __new__(cls, requires_grad=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        data = torch.empty(0, **factory_kwargs)
        return torch.Tensor._make_subclass(cls, data, requires_grad)

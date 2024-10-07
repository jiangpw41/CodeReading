'''
原始行数2603，3个class（核心是class module，2249行），10个函数
'''

# mypy: allow-untyped-defs
# 用于静态类型检查器 mypy 的指令（或标记），目的是告知 mypy 允许没有类型注解的函数定义，而不触发类型检查警告或错误。
from collections import OrderedDict, namedtuple
import itertools
import warnings
import functools
import weakref

import torch
from torch._prims_common import DeviceLikeType
from ..parameter import Parameter                       # 从上级目录导入Parameter这个对torch.Tensor的高级封装类
import torch.utils.hooks as hooks

from torch import Tensor, device, dtype
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List             # 导入了多种类型注解
from typing_extensions import Self
from ...utils.hooks import RemovableHandle
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

# 定义了模块公开的API列表，当使用from module import *这样的导入语句时，Python解释器将根据__all__列表来决定哪些名字是可见的，
# 即哪些名字将被导入到当前命名空间中。如果模块中没有定义__all__，那么所有的公共名字（不以下划线开头的名字）都将被导入。
__all__ = ['register_module_forward_pre_hook', 'register_module_forward_hook',
           'register_module_full_backward_pre_hook', 'register_module_backward_hook',
           'register_module_full_backward_hook', 'register_module_buffer_registration_hook',
           'register_module_module_registration_hook', 'register_module_parameter_registration_hook', 'Module']

# 定义了一个类型别名，表示梯度可以是一个张量元组或单个张量（Union表示一个值可以是多种类型中的一种）。
_grad_t = Union[Tuple[Tensor, ...], Tensor]


"""
请参阅 https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self 
用于使用“T”注释“self”。“Module”的许多方法返回“self”，我们希望这些返回值是子类的类型，而不是更松散的“Module”类型。
使用TypeVar定义了一个类型变量T，它被约束为Module的子类。这在泛型方法和返回类型中很有用，确保返回的是子类类型而不是更宽泛的Module类型

bound='Module'指定了类型变量T的上界（bound）。这意味着T可以是Module类型或者其任何子类的实例。在类型检查期间，T将被视为至少和Module一样具体的类型。
通俗来说：T的本质还是一个类型注释，只不过和其他确定的、专一的类型注释（int, float）不一样，T指向所有Module及其子类。
是多态的鲜活例子，允许一个函数能处理多种类型的数据
"""
T = TypeVar('T', bound='Module')

"""
本质是一个接收有意外的键的输出器，_IncompatibleKeys实例被用来存储和输出在某个过程中发现的不兼容的键（包括缺失的键和意外的键。）
如果所有的键都匹配，print(incompatible_keys)将输出<All keys matched successfully>。如果存在不匹配的键，它将输出这些键的列表。

重写了__repr__方法，以提供自定义的字符串表示。
"""
class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
    """
    namedtuple是collections模块中的一个工厂函数，用于创建一个继承自tuple的不可变数据结构，它拥有指定的字段名。这个数据结构通常称作命名元组（named tuple）。
    第一个参数'IncompatibleKeys'是命名元组的名称，它将作为类的名称。
    第二个参数['missing_keys', 'unexpected_keys']是一个列表，包含了命名元组中字段的名称。这些字段是命名元组的组成部分，可以像属性一样访问。
    """
    def __repr__(self):
        """
        __repr__ 是一个特殊方法，用于返回对象的官方字符串表示，通常用于调试。在这个类中，
        如果 missing_keys 和 unexpected_keys 列表都为空，表示所有的键都匹配成功，__repr__ 返回一个友好的消息 <All keys matched successfully>。
        如果有不匹配的键，__repr__ 调用父类（即 namedtuple 创建的元组子类）的 __repr__ 方法来返回默认的字符串表示。
        """
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super().__repr__()
    # __str__ 是另一个特殊方法，用于返回对象的非正式字符串表示，通常用于用户展示。
    __str__ = __repr__

# 用于给多行字符串的每一行添加指定数量空格缩进（除了第一行）的工具函数
def _addindent(s_, numSpaces):
    """
    s_：要添加缩进的原始字符串。
    numSpaces：要添加的空格数量。
    """
    s = s_.split('\n')          # 将输入的字符串s_按行分割成列表，每行作为列表的一个元素。
    # 如果原始字符串只有一行（即没有换行符），则不需要添加缩进，直接返回原始字符串。
    if len(s) == 1:
        return s_
    first = s.pop(0)            # 取出第一行，并从列表中移除。这样，first变量现在包含原始字符串的第一行，而列表s包含剩余的所有行。
    s = [(numSpaces * ' ') + line for line in s]        #  使用列表推导式为列表s中的每一行添加指定数量的空格。numSpaces * ' ' 创建一个由numSpaces个空格组成的字符串，然后将其与每一行连接。
    s = '\n'.join(s)            # 将添加了空格的行列表重新连接成一个多行字符串，行与行之间用换行符\n分隔。
    s = first + '\n' + s        # 将未添加空格的第一行first与添加了空格的剩余行s连接起来，保持原有的第一行不缩进，而从第二行开始添加缩进。
    return s


r"""This tracks hooks common to all modules that are executed immediately before
.registering the buffer/module/parameter
用于描述下面的代码。它说明了这些钩子是在注册 buffer/module/parameter **之前立即执行**的
涉及到模块（Module）的钩子（hook）机制。钩子是函数，可以在模块的生命周期中的特定点触发或调用。
下面三行：全局钩子字典，buffer/module/parameter三种类型的数据的注册钩子，都是有序字典（OrderedDict），使用整数作为键，可调用对象（Callable）作为值。
"""
_global_buffer_registration_hooks: Dict[int, Callable] = OrderedDict()
_global_module_registration_hooks: Dict[int, Callable] = OrderedDict()
_global_parameter_registration_hooks: Dict[int, Callable] = OrderedDict()

# 这个类封装Wrap了钩子函数，并提供了额外的功能，比如与特定模块的关联。
class _WrappedHook:
    """主要作用是提供一个钩子包装器，它可以存储关于钩子的附加信息（如是否与特定模块关联），并确保钩子在模块销毁后不会被错误地调用。
    这是PyTorch模块系统中钩子机制的一部分，允许开发者在模块的生命周期中插入自定义逻辑。"""
    def __init__(self, hook: Callable, module: Optional["Module"] = None):
        """
        构造函数接受一个钩子函数hook和一个可选的模块module。
        """
        self.hook: Callable = hook
        functools.update_wrapper(self, hook)            # 更新_WrappedHook实例的__dict__属性和__wrapped__属性，以保留原始钩子函数的元数据。

        self.with_module: bool = False                  # 指示钩子是否与特定模块关联。如果module为none则不关联，否则关联

        if module is not None:                          # 如果提供了module参数，使用weakref.ref创建钩子函数对模块的弱引用，这样就不会阻止模块被垃圾回收。
            self.module: weakref.ReferenceType[Module] = weakref.ref(module)
            self.with_module = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        这使得_WrappedHook实例可以像函数一样被调用。
        如果钩子与模块关联，并且Module存在，则调用钩子函数，并将模块作为第一个参数传递。
        """
        if self.with_module:
            module = self.module()
            if module is None:
                raise RuntimeError("You are trying to call the hook of a dead Module!")
            return self.hook(module, *args, **kwargs)
        return self.hook(*args, **kwargs)

    def __getstate__(self) -> Dict:
        """
        对象的序列化方法，用于在pickle过程中获取对象的状态。它返回一个包含钩子和相关状态的字典。
        """
        result = {"hook": self.hook, "with_module": self.with_module}
        if self.with_module:
            result["module"] = self.module()

        return result

    def __setstate__(self, state: Dict):
        """
        这是对象的反序列化方法，用于在unpickle过程中设置对象的状态。它根据传入的状态字典设置钩子和相关状态。
        """
        self.hook = state["hook"]
        self.with_module = state["with_module"]

        if self.with_module:
            if state["module"] is None:
                '如果尝试调用已销毁模块的钩子，或尝试恢复已销毁模块的钩子，将引发RuntimeError。'
                raise RuntimeError("You are trying to revive the hook of a dead Module!")
            self.module = weakref.ref(state["module"])


r"""This tracks hooks common to all modules that are executed before/after
calling forward and backward. This is global state used for debugging/profiling
purposes这跟踪在向前和向后调用之前/之后执行的所有模块的通用钩子。这是用于调试/分析目的的全局状态
这些钩子和状态管理机制允许开发者在模块的执行流程中插入自定义逻辑，例如：
（1）在计算图的构建或执行前后添加自定义操作。
（2）在反向传播时修改梯度。
（3）在前向传播前后执行特定的初始化或清理操作。"""
_global_backward_pre_hooks: Dict[int, Callable] = OrderedDict()          # 存储在后向传播开始之前执行的钩子。
_global_backward_hooks: Dict[int, Callable] = OrderedDict()              # 存储在后向传播过程中执行的钩子。
_global_is_full_backward_hook: Optional[bool] = None                     # 一个布尔值，指示是否注册了全后向钩子。全后向钩子会在每次反向传播时被调用。

_global_forward_pre_hooks: Dict[int, Callable] = OrderedDict()           # 存储在前向传播开始之前执行的钩子。
_global_forward_hooks: Dict[int, Callable] = OrderedDict()               # 存储在前向传播过程中执行的钩子。
_global_forward_hooks_always_called: Dict[int, bool] = OrderedDict()     # 存储一个布尔值，指示是否应该在每次前向传播时调用特定的前向钩子。

_EXTRA_STATE_KEY_SUFFIX = '_extra_state'        # 定义了一个额外状态键的后缀字符串。这通常用于模块的额外状态，例如在序列化和反序列化过程中存储和检索模块的额外信息。



# 下面八个钩子注册函数，都是在上述_global_buffer_registration_hooks等有序字典中注册钩子
# 并将其放入hooks.RemovableHandle中返回
def register_module_buffer_registration_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""注册所有模块通用的buffer注册hook.它接受一个参数 hook 和返回一个 RemovableHandle 对象

    .. warning ::

        这为nn.Module添加了全局状态。

    每次调用：`register_buffer`时都会调用钩子。它应该有以下签名signature(函数名以及其参数的集合)：
        hook(module, name, buffer) -> None or new buffer

    钩子可以修改输入，or return a single modified value in the hook.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """

    # 创建一个 RemovableHandle 对象，它与前面定义的 _global_buffer_registration_hooks 字典相关联。
    # RemovableHandle 类似于一个唯一标识符，可以用于后续移除或管理注册的钩子。
    handle = hooks.RemovableHandle(_global_buffer_registration_hooks)
    # 将传入的 hook 函数注册到 _global_buffer_registration_hooks 有序字典中。钩子将使用 handle.id 作为键，确保每个钩子都有一个唯一的标识。
    _global_buffer_registration_hooks[handle.id] = hook
    return handle  # 返回 RemovableHandle 对象。这个对象可以被用来引用或移除注册的钩子


def register_module_module_registration_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Register a module registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_module` is invoked.
    It should have the following signature::

        hook(module, name, submodule) -> None or new submodule

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = hooks.RemovableHandle(_global_module_registration_hooks)
    _global_module_registration_hooks[handle.id] = hook
    return handle


def register_module_parameter_registration_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Register a parameter registration hook common to all modules.

    .. warning ::

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_parameter` is invoked.
    It should have the following signature::

        hook(module, name, param) -> None or new parameter

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = hooks.RemovableHandle(_global_parameter_registration_hooks)
    _global_parameter_registration_hooks[handle.id] = hook
    return handle


def register_module_forward_pre_hook(hook: Callable[..., None]) -> RemovableHandle:
    r"""Register a forward pre-hook common to all modules.

    .. warning ::

        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time before :func:`forward` is invoked.
    It should have the following signature::

        hook(module, input) -> None or modified input

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    The hook can modify the input. User can either return a tuple or a
    single modified value in the hook. We will wrap the value into a tuple
    if a single value is returned(unless that value is already a tuple).

    This hook has precedence over the specific module hooks registered with
    ``register_forward_pre_hook``.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = hooks.RemovableHandle(_global_forward_pre_hooks)
    _global_forward_pre_hooks[handle.id] = hook
    return handle


def register_module_forward_hook(hook: Callable[..., None], *, always_call: bool = False) -> RemovableHandle:
    r"""Register a global forward hook for all the modules.

    .. warning ::

        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time after :func:`forward` has computed an output.
    It should have the following signature::

        hook(module, input, output) -> None or modified output

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    The hook can modify the output. It can modify the input inplace but
    it will not have effect on forward since this is called after
    :func:`forward` is called.

    Parameters:
        hook (Callable): The user defined hook to be registered.
        always_call (bool): If ``True`` the ``hook`` will be run regardless of
            whether an exception is raised while calling the Module.
            Default: ``False``
    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    This hook will be executed before specific module hooks registered with
    ``register_forward_hook``.
    """
    handle = hooks.RemovableHandle(_global_forward_hooks,
                                   extra_dict=_global_forward_hooks_always_called)
    _global_forward_hooks[handle.id] = hook
    if always_call:
        _global_forward_hooks_always_called[handle.id] = True
    return handle


def register_module_backward_hook(
    hook: Callable[['Module', _grad_t, _grad_t], Union[None, _grad_t]]
) -> RemovableHandle:
    r"""Register a backward hook common to all the modules.

    This function is deprecated in favor of
    :func:`torch.nn.modules.module.register_module_full_backward_hook`
    and the behavior of this function will change in future versions.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
    global _global_is_full_backward_hook
    if _global_is_full_backward_hook is True:
        raise RuntimeError("Cannot use both regular backward hooks and full backward hooks as a "
                           "global Module hook. Please use only one of them.")

    _global_is_full_backward_hook = False

    handle = hooks.RemovableHandle(_global_backward_hooks)
    _global_backward_hooks[handle.id] = hook
    return handle


def register_module_full_backward_pre_hook(
    hook: Callable[['Module', _grad_t], Union[None, _grad_t]]
) -> RemovableHandle:
    r"""Register a backward pre-hook common to all the modules.

    .. warning ::
        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    Hooks registered using this function behave in the same way as those
    registered by :meth:`torch.nn.Module.register_full_backward_pre_hook`.
    Refer to its documentation for more details.

    Hooks registered using this function will be called before hooks registered
    using :meth:`torch.nn.Module.register_full_backward_pre_hook`.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
    handle = hooks.RemovableHandle(_global_backward_pre_hooks)
    _global_backward_pre_hooks[handle.id] = hook
    return handle


def register_module_full_backward_hook(
    hook: Callable[['Module', _grad_t, _grad_t], Union[None, _grad_t]]
) -> RemovableHandle:
    r"""Register a backward hook common to all the modules.

    .. warning ::
        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    Hooks registered using this function behave in the same way as those
    registered by :meth:`torch.nn.Module.register_full_backward_hook`.
    Refer to its documentation for more details.

    Hooks registered using this function will be called before hooks registered
    using :meth:`torch.nn.Module.register_full_backward_hook`.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    """
    global _global_is_full_backward_hook
    if _global_is_full_backward_hook is False:
        raise RuntimeError("Cannot use both regular backward hooks and full backward hooks as a "
                           "global Module hook. Please use only one of them.")

    _global_is_full_backward_hook = True

    handle = hooks.RemovableHandle(_global_backward_hooks)
    _global_backward_hooks[handle.id] = hook
    return handle


# Module需要重写forward，如果没有，指出模块缺少必需的 forward 函数。
# 用于替代 PyTorch 模块中的 forward 函数。通过将forward定义为值而不是函数，让mypy不将反向规则应用于输入。
# Trick mypy into not applying contravariance rules to inputs by defining
# forward as a value, rather than a function.
# 另请参见https://github.com/python/mypy/issues/8795
def _forward_unimplemented(self, *input: Any) -> None:
    r"""定义每次调用时执行的计算。应被所有子类覆盖。

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
        如果 _forward_unimplemented 被调用，它将引发一个 NotImplementedError 异常，指出模块缺少必需的 forward 函数。
    """
    raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "forward" function')

"""
Module类作为所有神经网络模块的基类，内核有2：
（1）本身是功能模块，如Embedding，需要梯度的_parameters或者不需要梯度的buffer，
（2）本身是结构模块，如Model类，则保存相应的子模块的结构信息、state_dict
此外，定义了与前后向传播与状态字典相关的有序字典，并设置了以下功能：
（1）必须在子类中重写的forward函数
（2）对Module结构进行递归操作的apply方法
（3）模块参数移动方法to以及三个重载入口
（4）被调用call时采取的5种钩子和forward逻辑：__call__ : Callable[..., Any] = _wrapped_call_impl
（5）获取、修改属性
（6）state_dict相关方法
（7）param、module等子模块获取、tran/eval属性设置、数据并行、编译
"""
class Module:
    r"""所有神经网络模块的基类。
    你的模型也应该子类化这个类。
    模块还可以包含其他模块，允许将它们嵌套在树结构中。您可以将子模块指定为常规属性，例如：

        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
                
    以这种方式分配的子模块将被注册，并且当您调用：to方法时，module的的parameters也将被转换。
    ..注：
        根据上面的例子，在对子类进行赋值之前，必须对父类进行`__init__（）`调用。
    ：ivar training：布尔值，表示此模块是处于训练模式还是评估模式。
    """
    # 类属性两个
    dump_patches: bool = False          # 一个类属性，用于控制是否转储补丁（可能是为了调试目的）。
    _version: int = 1                   # module版本的类属性。
    r"""
    这允许更好地支持`load_state_dict`方法的向后兼容（BC）。在`load_state_dict`方法中，
    版本号version number将保存在返回的状态字典的属性`_metadata`中，因此会被缓存。
    `_metada`是一个字典，其键遵循状态字典的命名约定。有关如何在加载中使用此信息，请参阅`_load_from_state_dict `。”。

    如果在module中添加/删除了新的parameters/buffers，则应删除此数字，并且模块的`_load_from_state_dict`方法可以比较版本号，如果状态字典来自更改之前，则可以进行适当的更改。"""

    # 实例属性
    training: bool                                  # 指示模块是否处于训练模式（True）或评估模式（False）。
    _parameters: Dict[str, Optional[Parameter]]     # 一个字典，存储模块的参数，键是参数名，值是Parameter对象或None。
    _buffers: Dict[str, Optional[Tensor]]           # 一个字典，存储模块的缓冲区，键是缓冲区名，值是Tensor对象或None。
    _non_persistent_buffers_set: Set[str]           # 一个集合，包含非持久性缓冲区的名称，这些缓冲区在每次前向传播后不会被保存。

    # 前后向传播钩子
    _backward_pre_hooks: Dict[int, Callable]        # 字典：在反向传播开始前调用的钩子函数。
    _backward_hooks: Dict[int, Callable]            # 字典：在反向传播过程中调用的钩子函数。
    _is_full_backward_hook: Optional[bool]          # 布尔值，指示是否注册了全反向钩子。
    _forward_hooks: Dict[int, Callable]             # 字典：在前向传播过程中调用的钩子函数。 
    _forward_hooks_with_kwargs: Dict[int, bool]     # 字典：标记相应的_forward_hooks是否接受kwargs。由于JIT不支持Set[int]，因此将此dict用作一个集合，其中此dict中表示的所有钩子都接受kwargs。
    
    _forward_hooks_always_called: Dict[int, bool]   # 字典：即使引发异常，也终调用的前向钩子
    
    _forward_pre_hooks: Dict[int, Callable]         # 字典：在前向传播开始前调用的钩子函数。
    _forward_pre_hooks_with_kwargs: Dict[int, bool] # 标记_forward_pre_hooks中相应的钩子是否接受**kwargs参数。

    # 状态字典钩子
    _state_dict_hooks: Dict[int, Callable]          # 在模块状态字典（即参数和缓冲区）序列化之前调用的钩子函数。
    _load_state_dict_pre_hooks: Dict[int, Callable] # 在模块加载状态字典之前调用的钩子函数。
    _state_dict_pre_hooks: Dict[int, Callable]      # 在模块状态字典序列化之前调用的钩子函数。
    _load_state_dict_post_hooks: Dict[int, Callable]# 在模块加载状态字典之后调用的钩子函数。

    # 子模块
    _modules: Dict[str, Optional['Module']]         # 一个字典，存储模块的子模块，键是子模块名，值是Module对象。
    call_super_init: bool = False                   # 类属性，用于控制是否在子类的构造函数中调用父类的构造函数。在某些情况下，如果子类需要自定义初始化逻辑，可能会设置这个属性为False，以避免调用父类的__init__方法。
    _compiled_call_impl : Optional[Callable] = None # 用于存储编译后的调用实现。在PyTorch中，某些模块的操作可以通过TorchScript编译来提高性能。如果模块的方法被编译，_compiled_call_impl将是一个可调用对象，表示编译后的实现。如果未编译或不需要编译，这个属性将是None。

    def __init__(self, *args, **kwargs) -> None:
        """初始化内部模块状态，由nn.Module 和 ScriptModule共享
        接受任意数量的位置参数 *args 和关键字参数 **kwargs，并返回 None（即不返回任何值）。"""

        # 记录了 nn.Module 被使用的情况，这通常用于跟踪API的使用情况，以便进行性能监控或功能改进。
        torch._C._log_api_usage_once("python.nn_module")
        
        # 向后兼容性：call_super_init=False、且构造函数接收到了关键字参数 kwargs 或位置参数 args，将分别抛出 TypeError。
        # 这确保了当子类没有定义自己的 __init__ 方法或不希望处理额外的参数时，不会意外接收到参数。
        if self.call_super_init is False and bool(kwargs):
            raise TypeError(f"{type(self).__name__}.__init__() got an unexpected keyword argument '{next(iter(kwargs))}'"
                            "")

        if self.call_super_init is False and bool(args):
            raise TypeError(f"{type(self).__name__}.__init__() takes 1 positional argument but {len(args) + 1} were"
                            " given")

        """
        调用super().__setattr__('a', a)代替典型的self.a=a以避免Module.__setattr__的开销。
        Module的__setattr__对arameters, submodules, and buffers有特殊的处理，
        但对所有其他属性只是调用super（）__setattr_

        下面使用 super().__setattr__ 来初始化 Module 类的多个属性如training：布尔值，表示模块是否处于训练模式。
            各种以 _ 开头的属性，如 _parameters、_buffers 等，它们是有序字典（OrderedDict）或集合（set），用于存储模块的参数、缓冲区、钩子等。
        """
        super().__setattr__('training', True)
        super().__setattr__('_parameters', OrderedDict())
        super().__setattr__('_buffers', OrderedDict())
        super().__setattr__('_non_persistent_buffers_set', set())
        super().__setattr__('_backward_pre_hooks', OrderedDict())
        super().__setattr__('_backward_hooks', OrderedDict())
        super().__setattr__('_is_full_backward_hook', None)
        super().__setattr__('_forward_hooks', OrderedDict())
        super().__setattr__('_forward_hooks_with_kwargs', OrderedDict())
        super().__setattr__('_forward_hooks_always_called', OrderedDict())
        super().__setattr__('_forward_pre_hooks', OrderedDict())
        super().__setattr__('_forward_pre_hooks_with_kwargs', OrderedDict())
        super().__setattr__('_state_dict_hooks', OrderedDict())
        super().__setattr__('_state_dict_pre_hooks', OrderedDict())
        super().__setattr__('_load_state_dict_pre_hooks', OrderedDict())
        super().__setattr__('_load_state_dict_post_hooks', OrderedDict())
        super().__setattr__('_modules', OrderedDict())

        if self.call_super_init:
            "如果需要父类初始化，才初始化。这是一种安全的做法，确保如果父类有任何初始化逻辑，子类也会执行。"
            super().__init__(*args, **kwargs)

    # 提供了一个默认实现 _forward_unimplemented。这是一个未实现的前向传播方法，如果子类没有重写 forward 方法，将引发 NotImplementedError 异常。
    forward: Callable[..., Any] = _forward_unimplemented

    # 在module注册一个buffer
    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        r""".
        这通常用于注册不应被视为模型参数parameter的缓冲区buffer。
        例如，BatchNorm的`running_man`不是参数，而是模块状态的一部分。
        默认情况下，缓冲区是持久的，将与参数一起保存。
        可以通过将：attr:“persistent”设置为“False”来更改此行为。
        持久缓冲区和非持久缓冲区之间的唯一区别是，后者不会成为此模块的：attr:`state_dict`的一部分。

        缓冲区可以使用给定的名称作为属性进行访问，例如model.aaa_buffer

        Args:
            name (str): 缓冲区的名称。可以使用给定的名称从该模块访问缓冲区
            tensor (Tensor or None): 要注册的buffer的张量主体。如果为“None”，则在buffer上的一些操作如指定设备的'cuda'属性将被忽略；如果为“None”，则buffer不被包含在模块的state_dict属性中
            persistent (bool): 缓冲区是否是此模块的一部分：attr:`state_dict`
        Example::
            >>> self.register_buffer('running_mean', torch.zeros(num_features))
        """

        # 检查：ScriptModule不支持非持久性缓冲区
        if persistent is False and isinstance(self, torch.jit.ScriptModule):
            raise RuntimeError("ScriptModule does not support non-persistent buffers")

        # 检查_buffers属性是否在__dict__中，否则报错无法在模块之前分配缓冲区__init__（）调用
        if '_buffers' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError(f"buffer name should be a string. Got {torch.typename(name)}")
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute '{name}' already exists")
        elif tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError(f"cannot assign '{torch.typename(tensor)}' object to buffer '{name}' "
                            "(torch Tensor or None required)"
                            )
        else:
            "上述检查都没问题，则调用_global_buffer_registration_hooks字典中的hook函数 *递归* 处理传入的tensor"
            for hook in _global_buffer_registration_hooks.values():
                output = hook(self, name, tensor)
                if output is not None:
                    tensor = output
            self._buffers[name] = tensor
            if persistent:
                "如果是持久的，则从非持久列表中删除"
                self._non_persistent_buffers_set.discard(name)
            else:
                "否则加入"
                self._non_persistent_buffers_set.add(name)

    # 在module注册一个parameter，类上，递归用钩子函数操作完毕后加入Module的_parameters字典。
    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        r"""Add a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (str): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, str):
            raise TypeError(f"parameter name should be a string. Got {torch.typename(name)}")
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(f"cannot assign '{torch.typename(param)}' object to parameter '{name}' "
                            "(torch.nn.Parameter or None required)"
                            )
        elif param.grad_fn:
            raise ValueError(
                f"Cannot assign non-leaf Tensor to parameter '{name}'. Model "
                f"parameters must be created explicitly. To express '{name}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method.")
        else:
            for hook in _global_parameter_registration_hooks.values():
                output = hook(self, name, param)
                if output is not None:
                    param = output
            self._parameters[name] = param

    # 当Module实例作为结构上的父模块时：将子模块添加到当前模块。可以使用给定的名称作为属性访问该模块。主要对子模块的命名进行规范，
    # 然后递归调用_global_module_registration_hooks中的钩子，最后加入Module的_modules字典
    def add_module(self, name: str, module: Optional['Module']) -> None:
        r"""
        Args:
            name (str): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{torch.typename(module)} is not a Module subclass")
        elif not isinstance(name, str):
            raise TypeError(f"module name should be a string. Got {torch.typename(name)}")
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        elif '.' in name:
            raise KeyError(f"module name can't contain \".\", got: {name}")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        for hook in _global_module_registration_hooks.values():
            output = hook(self, name, module)
            if output is not None:
                module = output
        self._modules[name] = module

    # 与add_module一样
    def register_module(self, name: str, module: Optional['Module']) -> None:
        r"""Alias for :func:`add_module`."""
        self.add_module(name, module)

    # 根据子模块名检查是否存在该子模块，如果存在“target”给出的子模块，则返回该子模块，否则抛出错误。
    def get_submodule(self, target: str) -> "Module":
        """For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To check whether or not we have the ``linear`` submodule, we
        would call ``get_submodule("net_b.linear")``. To check whether
        we have the ``conv`` submodule, we would call
        ``get_submodule("net_b.net_c.conv")``.

        The runtime of ``get_submodule`` is bounded by the degree
        of module nesting in ``target``. A query against
        ``named_modules`` achieves the same result, but it is O(N) in
        the number of transitive modules. So, for a simple check to see
        if some submodule exists, ``get_submodule`` should always be
        used.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Module: The submodule referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Module``
        """
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        mod: torch.nn.Module = self

        for item in atoms:

            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no "
                                     "attribute `" + item + "`")

            mod = getattr(mod, item)

            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + item + "` is not "
                                     "an nn.Module")

        return mod

    # 同上，检查模型参数
    def get_parameter(self, target: str) -> "Parameter":
        """Return the parameter given by ``target`` if it exists, otherwise throw an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the Parameter
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Parameter: The Parameter referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Parameter``
        """
        module_path, _, param_name = target.rpartition(".")

        mod: torch.nn.Module = self.get_submodule(module_path)

        if not hasattr(mod, param_name):
            raise AttributeError(mod._get_name() + " has no attribute `"
                                 + param_name + "`")

        param: torch.nn.Parameter = getattr(mod, param_name)

        if not isinstance(param, torch.nn.Parameter):
            raise AttributeError("`" + param_name + "` is not an "
                                 "nn.Parameter")

        return param

    # 同上
    def get_buffer(self, target: str) -> "Tensor":
        """Return the buffer given by ``target`` if it exists, otherwise throw an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the buffer
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.Tensor: The buffer referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not a
                buffer
        """
        module_path, _, buffer_name = target.rpartition(".")

        mod: torch.nn.Module = self.get_submodule(module_path)

        if not hasattr(mod, buffer_name):
            raise AttributeError(mod._get_name() + " has no attribute `"
                                 + buffer_name + "`")

        buffer: torch.Tensor = getattr(mod, buffer_name)

        if buffer_name not in mod._buffers:
            raise AttributeError("`" + buffer_name + "` is not a buffer")

        return buffer

    # 返回任何额外的状态以包含在模块的state_dict中。不可用
    def get_extra_state(self) -> Any:
        """Return any extra state to include in the module's state_dict.

        Implement this and a corresponding :func:`set_extra_state` for your module
        if you need to store extra state. This function is called when building the
        module's `state_dict()`.

        Note that extra state should be picklable to ensure working serialization
        of the state_dict. We only provide provide backwards compatibility guarantees
        for serializing Tensors; other objects may break backwards compatibility if
        their serialized pickled form changes.

        Returns:
            object: Any extra state to store in the module's state_dict
        """
        raise RuntimeError(
            "Reached a code path in Module.get_extra_state() that should never be called. "
            "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
            "to report this bug.")

    # 同上，不可用
    def set_extra_state(self, state: Any) -> None:
        """Set extra state contained in the loaded `state_dict`.

        This function is called from :func:`load_state_dict` to handle any extra state
        found within the `state_dict`. Implement this function and a corresponding
        :func:`get_extra_state` for your module if you need to store extra state within its
        `state_dict`.

        Args:
            state (dict): Extra state from the `state_dict`
        """
        raise RuntimeError(
            "Reached a code path in Module.set_extra_state() that should never be called. "
            "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
            "to report this bug.")

    # 提供了一种灵活的方式来对模块张量执行原地操作
    def _apply(self, fn, recurse=True):
        """
        在 PyTorch 的 Module 类中是一个受保护的方法，用于对模块的所有参数和缓冲区应用给定的函数 fn。
        它通常用于对模块的张量执行原地操作，比如标准化或数据类型转换。以下是该方法的详细解释：
        接受两个参数：fn，要应用的函数；recurse，一个布尔标志，决定是否递归地对所有子模块应用该函数。
        
        如果 recurse 为 True，则使用 self.children() 遍历所有子模块，并对每个子模块使用 module._apply(fn) 应用函数。
        """
        if recurse:
            for module in self.children():      
                module._apply(fn)

        def compute_should_use_set_data(tensor, tensor_applied):
            """
            compute_should_use_set_data 是一个内部函数，根据兼容性和未来行为标志来决定是否对张量使用 .data 设置器。
            方法检查未来行为标志，如torch.__future__.get_overwrite_module_params_on_conversion() 和
              torch.__future__.get_swap_module_params_on_conversion()，这些标志控制参数如何更新
            """
            if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
                # 如果新张量与现有张量具有兼容的张量类型，则当前行为是使用`.data=`就地更改张量，
                # 未来行为是覆盖现有张量。然而，改变当前的行为是一种破坏BC的更改，我们希望它在未来的版本中发生。
                # 现在我们来介绍一下“torch.__future__.getoverwrite_module_params_onconversion（）`全局标志，
                # 让用户控制是否希望覆盖现有张量的未来行为。
                return not torch.__future__.get_overwrite_module_params_on_conversion()
            else:
                return False

        should_use_swap_tensors = torch.__future__.get_swap_module_params_on_conversion()

        for key, param in self._parameters.items():
            """
            遍历 self._parameters 中的所有参数。对于每个参数，它在 torch.no_grad() 环境中应用函数 fn，以避免跟踪自动求导历史。
            错误处理逻辑，确保在交换张量或更新参数过程中出现的任何问题都能被捕获并报告。
            """
            if param is None:
                continue
            # 存储在module中的张量是图叶子，我们不想跟踪“param_applied”的自动求导历史记录，所以我们必须使用“with torch.nograd（）”：`
            with torch.no_grad():
                param_applied = fn(param)

            # 方法要么使用 .data = 对参数进行原地更新，要么创建一个新的 Parameter 并替换 self._parameters 中的旧参数。
            p_should_use_set_data = compute_should_use_set_data(param, param_applied)

            # 子类可能有多个子张量，因此我们需要使用swap张量
            p_should_use_swap_tensors = should_use_swap_tensors or is_traceable_wrapper_subclass(param_applied)

            param_grad = param.grad     # 如果参数有梯度（param.grad），方法同样对梯度应用函数 fn，并相应地更新它，可以是原地更新，也可以是创建新的梯度。
            if p_should_use_swap_tensors:
                try:
                    if param_grad is not None:
                        # Accessing param.grad makes its at::Tensor's use_count 2, which will prevent swapping.
                        # Decrement use count of the gradient by setting to None
                        param.grad = None
                    param_applied = torch.nn.Parameter(param_applied, requires_grad=param.requires_grad)
                    torch.utils.swap_tensors(param, param_applied)
                except Exception as e:
                    if param_grad is not None:
                        param.grad = param_grad
                    raise RuntimeError(f"_apply(): Couldn't swap {self._get_name()}.{key}") from e
                out_param = param
            elif p_should_use_set_data:
                param.data = param_applied
                out_param = param
            else:
                assert isinstance(param, Parameter)
                assert param.is_leaf
                out_param = Parameter(param_applied, param.requires_grad)
                self._parameters[key] = out_param

            if param_grad is not None:
                with torch.no_grad():
                    grad_applied = fn(param_grad)
                g_should_use_set_data = compute_should_use_set_data(param_grad, grad_applied)
                if p_should_use_swap_tensors:
                    grad_applied.requires_grad_(param_grad.requires_grad)
                    try:
                        torch.utils.swap_tensors(param_grad, grad_applied)
                    except Exception as e:
                        raise RuntimeError(f"_apply(): Couldn't swap {self._get_name()}.{key}.grad") from e
                    out_param.grad = param_grad
                elif g_should_use_set_data:
                    assert out_param.grad is not None
                    out_param.grad.data = grad_applied
                else:
                    assert param_grad.is_leaf
                    out_param.grad = grad_applied.requires_grad_(param_grad.requires_grad)

        for key, buf in self._buffers.items():
            """
            遍历 self._buffers 中的所有缓冲区，并对其每个缓冲区应用函数 fn。"""
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self

    # 对自己或子模块应用函数
    def apply(self: T, fn: Callable[['Module'], None]) -> T:
        r"""递归地将`fn`应用于每个子模块（如`.children（）`返回的）以及self。
        典型用途包括初始化模型的参数
        (see also :ref:`nn-init-doc`).

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self

        Example::

            >>> @torch.no_grad()
            >>> def init_weights(m):
            >>>     print(m)
            >>>     if type(m) == nn.Linear:
            >>>         m.weight.fill_(1.0)
            >>>         print(m.weight)
            >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            >>> net.apply(init_weights)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.]], requires_grad=True)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.]], requires_grad=True)
            Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )

        """
        # 对自己或子模块应用函数
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    # 下面四个函数一个作用：
    # 将Module所有的parameters and buffers移动到GPU、ipu、xpu、cpu
    def cuda(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""这也使得相关参数和缓冲区成为不同的对象. So it should be called before constructing optimizer if the module will live on GPU while being optimized.
            因此，如果模块在优化时将驻留在GPU上，则应在构建优化器之前调用这个方法。
        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """

        # 调用_apply方法，传入的函数lambda 函数lambda t: t.cuda(device)。
        # 这个 lambda 函数对模块中的每个张量调用 .cuda(device) 方法，将其移动到 GPU。
        # t是张量，调用的是张量的cuda而不是module的cuda方法
        return self._apply(lambda t: t.cuda(device))

    def ipu(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""Move all model parameters and buffers to the IPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on IPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.ipu(device))

    def xpu(self: T, device: Optional[Union[int, device]] = None) -> T:
        r"""Move all model parameters and buffers to the XPU.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on XPU while being optimized.

        .. note::
            This method modifies the module in-place.

        Arguments:
            device (int, optional): if specified, all parameters will be
                copied to that device

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.xpu(device))

    def cpu(self: T) -> T:
        r"""Move all model parameters and buffers to the CPU.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.cpu())

    # 将所有参数和缓冲区强制转换`dst_type`数据类型
    def type(self: T, dst_type: Union[dtype, str]) -> T:
        r"""Casts all parameters and buffers to :attr:`dst_type`.

        .. note::
            This method modifies the module in-place.

        Args:
            dst_type (type or string): the desired type

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.type(dst_type))

    def float(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``float`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.float() if t.is_floating_point() else t)

    def double(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``double`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)

    def half(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``half`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)

    def bfloat16(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``bfloat16`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.bfloat16() if t.is_floating_point() else t)

    # 将模块的所有参数和缓冲区移动到指定的设备上，但是不复制存储（即不实际移动数据）。创建新的、空的张量，这些张量的形状和类型与原始张量相同，
    def to_empty(self: T, *, device: Optional[DeviceLikeType], recurse: bool = True) -> T:
        r"""Move the parameters and buffers to the specified device without copying storage.
        这个方法的一个可能用途是当你想改变张量的设备属性，但不想复制数据时。这在某些情况下可以节省内存和计算资源，特别是当你处理大型张量时。
        后续操作：如果你在调用 to_empty 之后立即对这些参数或缓冲区进行操作（例如，进行数学运算），那么 PyTorch 将自动处理数据的实际移动。
        也就是说，当你开始使用这些张量时，PyTorch 会确保它们在正确的设备上，并且如果需要，会将数据复制到那里。
        Args:
            device (:class:`torch.device`): The desired device of the parameters
                and buffers in this module.
            recurse (bool): Whether parameters and buffers of submodules should
                be recursively moved to the specified device.

        Returns:
            Module: self
        """
        return self._apply(lambda t: torch.empty_like(t, device=device), recurse=recurse)

    # 使用了 Python 的 @overload 装饰器来定义多个重载版本。这些重载提供了不同的调用方式，以满足不同的使用场景。
    # (1)允许用户指定设备和数据类型，以及一个可选的 non_blocking 参数，表示是否异步复制张量。如果 non_blocking 为 True，则复制操作不会阻塞当前线程。
    @overload
    def to(self, device: Optional[DeviceLikeType] = ..., dtype: Optional[dtype] = ...,
           non_blocking: bool = ...) -> Self:
        ...

    # (2)这个重载只接受数据类型和 non_blocking 参数，用于只更改张量的数据类型而不改变其所在的设备。
    @overload
    def to(self, dtype: dtype, non_blocking: bool = ...) -> Self:
        ...

    # (3)这个重载接受一个 Tensor 对象作为参数，用于将调用对象转换为与提供的 tensor 相同的数据类型和设备。
    @overload
    def to(self, tensor: Tensor, non_blocking: bool = ...) -> Self:
        ...

    # (4)实际的 to 方法接受任意数量的位置参数和关键字参数；移动和/或强制转换参数和缓冲区
    def to(self, *args, **kwargs):
        r"""Move and/or cast the parameters and buffers.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)
           :noindex:

        .. function:: to(dtype, non_blocking=False)
           :noindex:

        .. function:: to(tensor, non_blocking=False)
           :noindex:

        .. function:: to(memory_format=torch.channels_last)
           :noindex:

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point or complex :attr:`dtype`\ s. In addition, this method will
        only cast the floating point or complex parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
                the parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module
            memory_format (:class:`torch.memory_format`): the desired memory
                format for 4D parameters and buffers in this module (keyword
                only argument)

        Returns:
            Module: self

        Examples::

            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> linear = nn.Linear(2, 2)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]])
            >>> linear.to(torch.double)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]], dtype=torch.float64)
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
            >>> gpu1 = torch.device("cuda:1")
            >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
            >>> cpu = torch.device("cpu")
            >>> linear.to(cpu)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16)

            >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.3741+0.j,  0.2382+0.j],
                    [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
            >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
            tensor([[0.6122+0.j, 0.1150+0.j],
                    [0.6122+0.j, 0.1150+0.j],
                    [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

        """
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if dtype is not None:
            if not (dtype.is_floating_point or dtype.is_complex):
                raise TypeError('nn.Module.to only accepts floating point or complex '
                                f'dtypes, but got desired dtype={dtype}')
            if dtype.is_complex:
                warnings.warn(
                    "Complex modules are a new feature under active development whose design may change, "
                    "and some modules might not work as expected when using complex tensors as parameters or buffers. "
                    "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
                    "if a complex module does not work as expected.")

        def convert(t):
            try:
                if convert_to_format is not None and t.dim() in (4, 5):
                    return t.to(
                        device,
                        dtype if t.is_floating_point() or t.is_complex() else None,
                        non_blocking,
                        memory_format=convert_to_format,
                    )
                return t.to(
                    device,
                    dtype if t.is_floating_point() or t.is_complex() else None,
                    non_blocking,
                )
            except NotImplementedError as e:
                if str(e) == "Cannot copy out of meta tensor; no data!":
                    raise NotImplementedError(
                        f"{e} Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() "
                        f"when moving module from meta to a different device."
                    ) from None
                else:
                    raise

        return self._apply(convert)

    # 下面9个方法是关于注册、获取Module钩子以及检查状态的函数
    # (1)注册在完整反向传播开始之前的钩子
    def register_full_backward_pre_hook(
        self,
        hook: Callable[["Module", _grad_t], Union[None, _grad_t]],
        prepend: bool = False,
    ) -> RemovableHandle:
        r"""Register a backward pre-hook on the module.

        The hook will be called every time the gradients for the module are computed.
        The hook should have the following signature::

            hook(module, grad_output) -> tuple[Tensor] or None

        The :attr:`grad_output` is a tuple. The hook should
        not modify its arguments, but it can optionally return a new gradient with
        respect to the output that will be used in place of :attr:`grad_output` in
        subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
        all non-Tensor arguments.

        For technical reasons, when this hook is applied to a Module, its forward function will
        receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
        of each Tensor returned by the Module's forward function.

        .. warning ::
            Modifying inputs inplace is not allowed when using backward hooks and
            will raise an error.

        Args:
            hook (Callable): The user-defined hook to be registered.
            prepend (bool): If true, the provided ``hook`` will be fired before
                all existing ``backward_pre`` hooks on this
                :class:`torch.nn.modules.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``backward_pre`` hooks
                on this :class:`torch.nn.modules.Module`. Note that global
                ``backward_pre`` hooks registered with
                :func:`register_module_full_backward_pre_hook` will fire before
                all hooks registered by this method.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

        """
        handle = hooks.RemovableHandle(self._backward_pre_hooks)
        self._backward_pre_hooks[handle.id] = hook
        if prepend:
            '表示是否将新注册的钩子函数添加到钩子列表的开头（即反向传播时将首先执行这个钩子）。'
            self._backward_pre_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle

    # (2)注册在反向传播时的钩子，不论是否是完整反向传播。
    def register_backward_hook(
        self, hook: Callable[['Module', _grad_t, _grad_t], Union[None, _grad_t]]
    ) -> RemovableHandle:
        r"""Register a backward hook on the module.

        This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
        the behavior of this function will change in future versions.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

        """
        if self._is_full_backward_hook is True:
            raise RuntimeError("Cannot use both regular backward hooks and full backward hooks on a "
                               "single Module. Please use only one of them.")

        self._is_full_backward_hook = False

        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    # (3)在完整反向传播结束之后执行的钩子函数。
    def register_full_backward_hook(
        self,
        hook: Callable[["Module", _grad_t, _grad_t], Union[None, _grad_t]],
        prepend: bool = False,
    ) -> RemovableHandle:
        r"""Register a backward hook on the module.

        The hook will be called every time the gradients with respect to a module
        are computed, i.e. the hook will execute if and only if the gradients with
        respect to module outputs are computed. The hook should have the following
        signature::

            hook(module, grad_input, grad_output) -> tuple(Tensor) or None

        The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
        with respect to the inputs and outputs respectively. The hook should
        not modify its arguments, but it can optionally return a new gradient with
        respect to the input that will be used in place of :attr:`grad_input` in
        subsequent computations. :attr:`grad_input` will only correspond to the inputs given
        as positional arguments and all kwarg arguments are ignored. Entries
        in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
        arguments.

        For technical reasons, when this hook is applied to a Module, its forward function will
        receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
        of each Tensor returned by the Module's forward function.

        .. warning ::
            Modifying inputs or outputs inplace is not allowed when using backward hooks and
            will raise an error.

        Args:
            hook (Callable): The user-defined hook to be registered.
            prepend (bool): If true, the provided ``hook`` will be fired before
                all existing ``backward`` hooks on this
                :class:`torch.nn.modules.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``backward`` hooks on
                this :class:`torch.nn.modules.Module`. Note that global
                ``backward`` hooks registered with
                :func:`register_module_full_backward_hook` will fire before
                all hooks registered by this method.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``

        """
        if self._is_full_backward_hook is False:
            raise RuntimeError("Cannot use both regular backward hooks and full backward hooks on a "
                               "single Module. Please use only one of them.")

        self._is_full_backward_hook = True

        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        if prepend:
            self._backward_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle

    # 获取已有反向传播钩子
    def _get_backward_hooks(self):
        r"""Return the backward hooks for use in the call function.

        It returns two lists, one with the full backward hooks and one with the non-full
        backward hooks.
        """
        full_backward_hooks: List[Callable] = []
        if (_global_is_full_backward_hook is True):
            full_backward_hooks += _global_backward_hooks.values()
        if (self._is_full_backward_hook is True):
            full_backward_hooks += self._backward_hooks.values()

        non_full_backward_hooks: List[Callable] = []
        if (_global_is_full_backward_hook is False):
            non_full_backward_hooks += _global_backward_hooks.values()
        if (self._is_full_backward_hook is False):
            non_full_backward_hooks += self._backward_hooks.values()

        return full_backward_hooks, non_full_backward_hooks

    def _get_backward_pre_hooks(self):
        backward_pre_hooks: List[Callable] = []
        backward_pre_hooks += _global_backward_pre_hooks.values()
        backward_pre_hooks += self._backward_pre_hooks.values()

        return backward_pre_hooks

    def _maybe_warn_non_full_backward_hook(self, inputs, result, grad_fn):
        if not isinstance(result, torch.Tensor):
            if not (isinstance(result, tuple) and all(isinstance(r, torch.Tensor) for r in result)):
                warnings.warn(
                    "Using non-full backward hooks on a Module that does not return a "
                    "single Tensor or a tuple of Tensors is deprecated and will be removed "
                    "in future versions. This hook will be missing some of the grad_output. "
                    "Please use register_full_backward_hook to get the documented behavior.",
                    FutureWarning,
                    stacklevel=2,
                )
                return
        else:
            result = (result,)

        if not isinstance(inputs, torch.Tensor):
            if not (isinstance(inputs, tuple) and all(isinstance(i, torch.Tensor) for i in inputs)):
                warnings.warn(
                    "Using non-full backward hooks on a Module that does not take as input a "
                    "single Tensor or a tuple of Tensors is deprecated and will be removed "
                    "in future versions. This hook will be missing some of the grad_input. "
                    "Please use register_full_backward_hook to get the documented behavior.",
                    FutureWarning,
                    stacklevel=2,
                )
                return
        else:
            inputs = (inputs,)

        # At this point we are sure that inputs and result are tuple of Tensors
        out_grad_fn = {r.grad_fn for r in result if r.grad_fn is not None}
        if len(out_grad_fn) == 0 or (len(out_grad_fn) == 1 and grad_fn not in out_grad_fn):
            warnings.warn(
                "Using a non-full backward hook when outputs are nested in python data structure "
                "is deprecated and will be removed in future versions. This hook will be missing "
                "some grad_output.",
                FutureWarning,
                stacklevel=2,
            )
        elif len(out_grad_fn) > 1:
            warnings.warn(
                "Using a non-full backward hook when outputs are generated by different autograd Nodes "
                "is deprecated and will be removed in future versions. This hook will be missing "
                "some grad_output. Please use register_full_backward_hook to get the documented behavior.",
                FutureWarning,
                stacklevel=2,
            )
        else:
            # At this point the grad_output part of the hook will most likely be correct
            inputs_grad_fn = {i.grad_fn for i in inputs if i.grad_fn is not None}

            next_functions = {n[0] for n in grad_fn.next_functions}

            if inputs_grad_fn != next_functions:
                warnings.warn(
                    "Using a non-full backward hook when the forward contains multiple autograd Nodes "
                    "is deprecated and will be removed in future versions. This hook will be missing "
                    "some grad_input. Please use register_full_backward_hook to get the documented "
                    "behavior.",
                    FutureWarning,
                    stacklevel=2,
                )

    # (4)在前向传播之前的钩子
    def register_forward_pre_hook(
        self,
        hook: Union[
            Callable[[T, Tuple[Any, ...]], Optional[Any]],
            Callable[[T, Tuple[Any, ...], Dict[str, Any]], Optional[Tuple[Any, Dict[str, Any]]]],
        ],
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> RemovableHandle:
        r"""Register a forward pre-hook on the module.

        The hook will be called every time before :func:`forward` is invoked.


        If ``with_kwargs`` is false or not specified, the input contains only
        the positional arguments given to the module. Keyword arguments won't be
        passed to the hooks and only to the ``forward``. The hook can modify the
        input. User can either return a tuple or a single modified value in the
        hook. We will wrap the value into a tuple if a single value is returned
        (unless that value is already a tuple). The hook should have the
        following signature::

            hook(module, args) -> None or modified input

        If ``with_kwargs`` is true, the forward pre-hook will be passed the
        kwargs given to the forward function. And if the hook modifies the
        input, both the args and kwargs should be returned. The hook should have
        the following signature::

            hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If true, the provided ``hook`` will be fired before
                all existing ``forward_pre`` hooks on this
                :class:`torch.nn.modules.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``forward_pre`` hooks
                on this :class:`torch.nn.modules.Module`. Note that global
                ``forward_pre`` hooks registered with
                :func:`register_module_forward_pre_hook` will fire before all
                hooks registered by this method.
                Default: ``False``
            with_kwargs (bool): If true, the ``hook`` will be passed the kwargs
                given to the forward function.
                Default: ``False``

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(
            self._forward_pre_hooks,
            extra_dict=self._forward_pre_hooks_with_kwargs
        )
        self._forward_pre_hooks[handle.id] = hook
        if with_kwargs:
            self._forward_pre_hooks_with_kwargs[handle.id] = True

        if prepend:
            self._forward_pre_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle

    # (5)在前向传播时的钩子
    def register_forward_hook(
        self,
        hook: Union[
            Callable[[T, Tuple[Any, ...], Any], Optional[Any]],
            Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]],
        ],
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
    ) -> RemovableHandle:
        r"""Register a forward hook on the module.

        The hook will be called every time after :func:`forward` has computed an output.

        If ``with_kwargs`` is ``False`` or not specified, the input contains only
        the positional arguments given to the module. Keyword arguments won't be
        passed to the hooks and only to the ``forward``. The hook can modify the
        output. It can modify the input inplace but it will not have effect on
        forward since this is called after :func:`forward` is called. The hook
        should have the following signature::

            hook(module, args, output) -> None or modified output

        If ``with_kwargs`` is ``True``, the forward hook will be passed the
        ``kwargs`` given to the forward function and be expected to return the
        output possibly modified. The hook should have the following signature::

            hook(module, args, kwargs, output) -> None or modified output

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If ``True``, the provided ``hook`` will be fired
                before all existing ``forward`` hooks on this
                :class:`torch.nn.modules.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``forward`` hooks on
                this :class:`torch.nn.modules.Module`. Note that global
                ``forward`` hooks registered with
                :func:`register_module_forward_hook` will fire before all hooks
                registered by this method.
                Default: ``False``
            with_kwargs (bool): If ``True``, the ``hook`` will be passed the
                kwargs given to the forward function.
                Default: ``False``
            always_call (bool): If ``True`` the ``hook`` will be run regardless of
                whether an exception is raised while calling the Module.
                Default: ``False``

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(
            self._forward_hooks,
            extra_dict=[self._forward_hooks_with_kwargs, self._forward_hooks_always_called],
        )
        self._forward_hooks[handle.id] = hook
        if with_kwargs:
            self._forward_hooks_with_kwargs[handle.id] = True
        if always_call:
            self._forward_hooks_always_called[handle.id] = True
        if prepend:
            self._forward_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle

    #　前向传播时，如果模块的 forward 函数已经被 JIT 编译，就使用编译过的版本；如果没有被编译，则直接执行 Python 层面的 forward 函数。
    def _slow_forward(self, *input, **kwargs):
        """
        这个方法的实现确保了在TorchScript追踪过程中，模块的前向传播能够正确地记录作用域信息，
        同时也支持在没有追踪的情况下直接执行前向传播。这是PyTorch中TorchScript功能的一部分，
        允许开发者将Python代码转换为TorchScript，以提高性能和可移植性。
        """
        tracing_state = torch._C._get_tracing_state()
        if not tracing_state or isinstance(self.forward, torch._C.ScriptMethod):
            return self.forward(*input, **kwargs)
        recording_scopes = torch.jit._trace._trace_module_map is not None
        if recording_scopes:
            # type ignore was added because at this point one knows that
            # torch.jit._trace._trace_module_map is not Optional and has type Dict[Any, Any]
            name = torch.jit._trace._trace_module_map[self] if self in torch.jit._trace._trace_module_map else None  # type: ignore[index, operator] # noqa: B950
            if name:
                tracing_state.push_scope(name)
            else:
                recording_scopes = False
        try:
            result = self.forward(*input, **kwargs)
        finally:
            if recording_scopes:
                tracing_state.pop_scope()
        return result

    # 作为模块调用的包装器（wrapper），用于决定是执行 JIT 编译过的前向传播实现，还是执行普通的 Python 前向传播实现。调用下者
    def _wrapped_call_impl(self, *args, **kwargs):
        if self._compiled_call_impl is not None:
            return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
        else:
            return self._call_impl(*args, **kwargs)
    
    # 是模块调用的实际实现，它处理了模块的前向传播，包括执行前向钩子（forward hooks）、前向传播本身，以及设置反向钩子（backward hooks）
    def _call_impl(self, *args, **kwargs):

        # 根据是否处于TorchScript追踪状态，选择使用 _slow_forward 方法还是直接使用 self.forward。
        forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)

        # 如果没有注册任何钩子（四种钩子），直接调用前向调用并返回结果。
        if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
                or _global_backward_pre_hooks or _global_backward_hooks
                or _global_forward_hooks or _global_forward_pre_hooks):
            return forward_call(*args, **kwargs)

        # 如果有钩子注册，进入try模块
        try:
            result = None
            called_always_called_hooks = set()

            full_backward_hooks, non_full_backward_hooks = [], []
            backward_pre_hooks = []
            if self._backward_pre_hooks or _global_backward_pre_hooks:
                backward_pre_hooks = self._get_backward_pre_hooks()

            # 获取将被调用的反向钩子，分为全反向钩子和非全反向钩子。
            if self._backward_hooks or _global_backward_hooks:
                full_backward_hooks, non_full_backward_hooks = self._get_backward_hooks()

            # 如果注册了前向预钩子，遍历并执行它们，可能修改 args 和 kwargs。
            if _global_forward_pre_hooks or self._forward_pre_hooks:
                for hook_id, hook in (
                    *_global_forward_pre_hooks.items(),
                    *self._forward_pre_hooks.items(),
                ):
                    if hook_id in self._forward_pre_hooks_with_kwargs:
                        args_kwargs_result = hook(self, args, kwargs)  # type: ignore[misc]
                        if args_kwargs_result is not None:
                            if isinstance(args_kwargs_result, tuple) and len(args_kwargs_result) == 2:
                                args, kwargs = args_kwargs_result
                            else:
                                raise RuntimeError(
                                    "forward pre-hook must return None or a tuple "
                                    f"of (new_args, new_kwargs), but got {args_kwargs_result}."
                                )
                    else:
                        args_result = hook(self, args)
                        if args_result is not None:
                            if not isinstance(args_result, tuple):
                                args_result = (args_result,)
                            args = args_result

            # 如果存在反向预钩子或全反向钩子，创建一个 BackwardHook 对象，并设置输入钩子。
            bw_hook = None
            if full_backward_hooks or backward_pre_hooks:
                bw_hook = hooks.BackwardHook(self, full_backward_hooks, backward_pre_hooks)
                args = bw_hook.setup_input_hook(args)

            # 调用前向调用并获取结果。
            result = forward_call(*args, **kwargs)

            # 如果注册了前向钩子，遍历并执行它们，可能修改结果。
            if _global_forward_hooks or self._forward_hooks:
                for hook_id, hook in (
                    *_global_forward_hooks.items(),
                    *self._forward_hooks.items(),
                ):
                    # mark that always called hook is run
                    if hook_id in self._forward_hooks_always_called or hook_id in _global_forward_hooks_always_called:
                        called_always_called_hooks.add(hook_id)

                    if hook_id in self._forward_hooks_with_kwargs:
                        hook_result = hook(self, args, kwargs, result)
                    else:
                        hook_result = hook(self, args, result)

                    if hook_result is not None:
                        result = hook_result

            # 如果存在反向钩子，将结果传递给反向钩子的输出设置。
            if bw_hook:
                if not isinstance(result, (torch.Tensor, tuple)):
                    warnings.warn("For backward hooks to be called,"
                                  " module output should be a Tensor or a tuple of Tensors"
                                  f" but received {type(result)}")
                result = bw_hook.setup_output_hook(result)

            # 对于非全反向钩子，找到结果中的 Tensor 或 Tensor 元组，并为它们的 grad_fn 注册钩子。Handle the non-full backward hooks
            if non_full_backward_hooks:
                var = result
                while not isinstance(var, torch.Tensor):
                    if isinstance(var, dict):
                        var = next(v for v in var.values() if isinstance(v, torch.Tensor))
                    else:
                        var = var[0]
                grad_fn = var.grad_fn
                if grad_fn is not None:
                    for hook in non_full_backward_hooks:
                        grad_fn.register_hook(_WrappedHook(hook, self))
                    self._maybe_warn_non_full_backward_hook(args, result, grad_fn)

            return result

        except Exception:
            # run如果尚未运行，则始终调用钩子。目前只有前向钩子具有always_call选项，但也许这个功能也应该添加到完整的后向钩子中。
            for hook_id, hook in _global_forward_hooks.items():
                if hook_id in _global_forward_hooks_always_called and hook_id not in called_always_called_hooks:  # type: ignore[possibly-undefined]
                    try:
                        hook_result = hook(self, args, result)  # type: ignore[possibly-undefined]
                        if hook_result is not None:
                            result = hook_result
                    except Exception as e:
                        warnings.warn("global module forward hook with ``always_call=True`` raised an exception "
                                      f"that was silenced as another error was raised in forward: {str(e)}")
                        continue

            for hook_id, hook in self._forward_hooks.items():
                if hook_id in self._forward_hooks_always_called and hook_id not in called_always_called_hooks:  # type: ignore[possibly-undefined]
                    try:
                        if hook_id in self._forward_hooks_with_kwargs:
                            hook_result = hook(self, args, kwargs, result)  # type: ignore[possibly-undefined]
                        else:
                            hook_result = hook(self, args, result)  # type: ignore[possibly-undefined]
                        if hook_result is not None:
                            result = hook_result
                    except Exception as e:
                        warnings.warn("module forward hook with ``always_call=True`` raised an exception "
                                      f"that was silenced as another error was raised in forward: {str(e)}")
                        continue
            # raise exception raised in try block
            raise
    
    # __call__方法是一个特殊方法，用于让一个类实例表现得像函数一样，可以被调用。当定义了一个__call__方法后，类的实例可以通过()语法执行。
    __call__ : Callable[..., Any] = _wrapped_call_impl


    # 下面两个是PyTorch Module 类中用于序列化和反序列化的两个特殊方法
    # 定义了当对象需要被序列化时的行为：复制对象的字典属性，这个字典包含了类实例的所有属性和值。
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_compiled_call_impl", None)
        return state
    # 定义了当对象从序列化状态恢复（反序列化）时的行为。
    def __setstate__(self, state):
        self.__dict__.update(state)

        # Support loading old checkpoints that don't have the following attrs:
        if '_forward_pre_hooks' not in self.__dict__:
            self._forward_pre_hooks = OrderedDict()
        if '_forward_pre_hooks_with_kwargs' not in self.__dict__:
            self._forward_pre_hooks_with_kwargs = OrderedDict()
        if '_forward_hooks_with_kwargs' not in self.__dict__:
            self._forward_hooks_with_kwargs = OrderedDict()
        if '_forward_hooks_always_called' not in self.__dict__:
            self._forward_hooks_always_called = OrderedDict()
        if '_state_dict_hooks' not in self.__dict__:
            self._state_dict_hooks = OrderedDict()
        if '_state_dict_pre_hooks' not in self.__dict__:
            self._state_dict_pre_hooks = OrderedDict()
        if '_load_state_dict_pre_hooks' not in self.__dict__:
            self._load_state_dict_pre_hooks = OrderedDict()
        if '_load_state_dict_post_hooks' not in self.__dict__:
            self._load_state_dict_post_hooks = OrderedDict()
        if '_non_persistent_buffers_set' not in self.__dict__:
            self._non_persistent_buffers_set = set()
        if '_is_full_backward_hook' not in self.__dict__:
            self._is_full_backward_hook = None
        if '_backward_pre_hooks' not in self.__dict__:
            self._backward_pre_hooks = OrderedDict()


    # 下面三个是关于属性获取、设置、删除的函数

    # 关于返回类型：我们选择在`__getatr__`类型签名中返回`Any`，而不是更严格的`Union[Tensor，Module]。这
    # 样做是为了更好地与最终用户的各种类型检查器进行互操作。
    # 使用更严格的返回类型并不能很好地与`register_buffer（）`配合使用，并迫使人们过度使用类型忽略、断言、强制转换等。
    # 请参阅此处关于返回`Union `问题的完整讨论https://github.com/microsoft/pyright/issues/4213On the return type:
    def __getattr__(self, name: str) -> Any:
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(f"cannot assign '{torch.typename(value)}' as parameter '{name}' "
                                "(torch.nn.Parameter or None expected)"
                                )
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(f"cannot assign '{torch.typename(value)}' as child module '{name}' "
                                    "(torch.nn.Module or None expected)"
                                    )
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, torch.Tensor):
                        raise TypeError(f"cannot assign '{torch.typename(value)}' as buffer '{name}' "
                                        "(torch.Tensor or None expected)"
                                        )
                    for hook in _global_buffer_registration_hooks.values():
                        output = hook(self, name, value)
                        if output is not None:
                            value = output
                    buffers[name] = value
                else:
                    super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

    # 注册state_dict（之前）的两个钩子函数，以及保存为state_dict的方法
    def _register_state_dict_hook(self, hook):
        r"""Register a state-dict hook.

        These hooks will be called with arguments: `self`, `state_dict`,
        `prefix`, `local_metadata`, after the `state_dict` of `self` is set.
        Note that only parameters and buffers of `self` or its children are
        guaranteed to exist in `state_dict`. The hooks may modify `state_dict`
        inplace or return a new one.
        """
        handle = hooks.RemovableHandle(self._state_dict_hooks)
        self._state_dict_hooks[handle.id] = hook
        return handle

    def register_state_dict_pre_hook(self, hook):
        r"""Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

        These hooks will be called with arguments: ``self``, ``prefix``,
        and ``keep_vars`` before calling ``state_dict`` on ``self``. The registered
        hooks can be used to perform pre-processing before the ``state_dict``
        call is made.
        """
        handle = hooks.RemovableHandle(self._state_dict_pre_hooks)
        self._state_dict_pre_hooks[handle.id] = hook
        return handle

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Save module state to the `destination` dictionary.

        The `destination` dictionary will contain the state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else buf.detach()
        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if getattr(self.__class__, "get_extra_state", Module.get_extra_state) is not Module.get_extra_state:
            destination[extra_state_key] = self.get_extra_state()

    # The user can pass an optional arbitrary mappable object to `state_dict`, in which case `state_dict` returns
    # back that same object. But if they pass nothing, an `OrderedDict` is created and returned.
    T_destination = TypeVar('T_destination', bound=Dict[str, Any])

    # state_dict的两个重载和一个本体
    @overload
    def state_dict(self, *, destination: T_destination, prefix: str = ..., keep_vars: bool = ...) -> T_destination:
        ...

    @overload
    def state_dict(self, *, prefix: str = ..., keep_vars: bool = ...) -> Dict[str, Any]:
        ...

    # TODO: Change `*args` to `*` and remove the corresponding warning in docs when BC allows.
    # Also remove the logic for arg parsing together.
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        r"""Return a dictionary containing references to the whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.
        Parameters and buffers set to ``None`` are not included.

        .. note::
            The returned object is a shallow copy. It contains references
            to the module's parameters and buffers.

        .. warning::
            Currently ``state_dict()`` also accepts positional arguments for
            ``destination``, ``prefix`` and ``keep_vars`` in order. However,
            this is being deprecated and keyword arguments will be enforced in
            future releases.

        .. warning::
            Please avoid the use of argument ``destination`` as it is not
            designed for end-users.

        Args:
            destination (dict, optional): If provided, the state of module will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            prefix (str, optional): a prefix added to parameter and buffer
                names to compose the keys in state_dict. Default: ``''``.
            keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
                returned in the state dict are detached from autograd. If it's
                set to ``True``, detaching will not be performed.
                Default: ``False``.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> module.state_dict().keys()
            ['bias', 'weight']

        """
        # TODO: Remove `args` and the parsing logic when BC allows.
        if len(args) > 0:
            # DeprecationWarning is ignored by default
            warnings.warn(
                "Positional args are being deprecated, use kwargs instead. Refer to "
                "https://pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.state_dict"
                " for details.",
                FutureWarning,
                stacklevel=2,
            )
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == '':
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        for hook in self._state_dict_pre_hooks.values():
            hook(self, prefix, keep_vars)
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    # 注册加载state_dict（pre）的两个hook
    def _register_load_state_dict_pre_hook(self, hook, with_module=False):
        r"""Register a pre-hook for the :meth:`~torch.nn.Module.load_state_dict` method.

        These hooks will be called with arguments: `state_dict`, `prefix`,
        `local_metadata`, `strict`, `missing_keys`, `unexpected_keys`,
        `error_msgs`, before loading `state_dict` into `self`. These arguments
        are exactly the same as those of `_load_from_state_dict`.

        If ``with_module`` is ``True``, then the first argument to the hook is
        an instance of the module.

        Arguments:
            hook (Callable): Callable hook that will be invoked before
                loading the state dict.
            with_module (bool, optional): Whether or not to pass the module
                instance to the hook as the first parameter.
        """
        handle = hooks.RemovableHandle(self._load_state_dict_pre_hooks)
        self._load_state_dict_pre_hooks[handle.id] = _WrappedHook(hook, self if with_module else None)
        return handle

    def register_load_state_dict_post_hook(self, hook):
        r"""Register a post hook to be run after module's ``load_state_dict`` is called.

        It should have the following signature::
            hook(module, incompatible_keys) -> None

        The ``module`` argument is the current module that this hook is registered
        on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
        of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
        is a ``list`` of ``str`` containing the missing keys and
        ``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

        The given incompatible_keys can be modified inplace if needed.

        Note that the checks performed when calling :func:`load_state_dict` with
        ``strict=True`` are affected by modifications the hook makes to
        ``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
        set of keys will result in an error being thrown when ``strict=True``, and
        clearing out both missing and unexpected keys will avoid an error.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._load_state_dict_post_hooks)
        self._load_state_dict_post_hooks[handle.id] = hook
        return handle

    # 从state_dict加载
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        r"""Copy parameters and buffers from :attr:`state_dict` into only this module, but not its descendants.

        This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.
        Additionally, :attr:`local_metadata` can also contain the key
        `assign_to_params_buffers` that indicates whether keys should be
        assigned their corresponding tensor in the state_dict.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}
        assign_to_params_buffers = local_metadata.get("assign_to_params_buffers", False)
        use_swap_tensors = torch.__future__.get_swap_module_params_on_conversion()

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if not torch.overrides.is_tensor_like(input_param):
                    error_msgs.append(f'While copying the parameter named "{key}", '
                                      'expected torch.Tensor or Tensor-like object from checkpoint but '
                                      f'received {type(input_param)}'
                                      )
                    continue

                # This is used to avoid copying uninitialized parameters into
                # non-lazy modules, since they dont have the hook to do the checks
                # in such case, it will error when accessing the .shape attribute.
                is_param_lazy = torch.nn.parameter.is_lazy(param)
                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if not is_param_lazy and len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if not is_param_lazy and input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append(f'size mismatch for {key}: copying a param with shape {input_param.shape} from checkpoint, '
                                      f'the shape in current model is {param.shape}.')
                    continue

                if param.is_meta and not input_param.is_meta and not assign_to_params_buffers:
                    warnings.warn(f'for {key}: copying from a non-meta parameter in the checkpoint to a meta '
                                  'parameter in the current model, which is a no-op. (Did you mean to '
                                  'pass `assign=True` to assign items in the state dictionary to their '
                                  'corresponding key in the module instead of copying them in place?)')

                try:
                    with torch.no_grad():
                        if use_swap_tensors:
                            new_input_param = param.module_load(input_param, assign=assign_to_params_buffers)
                            if id(new_input_param) == id(input_param) or id(new_input_param) == id(param):
                                raise RuntimeError("module_load returned one of self or other, please .detach() "
                                                   "the result if returning one of the inputs in module_load")
                            if (isinstance(param, torch.nn.Parameter)):
                                if not isinstance(new_input_param, torch.nn.Parameter):
                                    new_input_param = torch.nn.Parameter(new_input_param, requires_grad=param.requires_grad)
                                else:
                                    new_input_param.requires_grad_(param.requires_grad)
                            torch.utils.swap_tensors(param, new_input_param)
                            del new_input_param
                        elif assign_to_params_buffers:
                            # Shape checks are already done above
                            if (isinstance(param, torch.nn.Parameter)):
                                if not isinstance(input_param, torch.nn.Parameter):
                                    input_param = torch.nn.Parameter(input_param, requires_grad=param.requires_grad)
                                else:
                                    input_param.requires_grad_(param.requires_grad)
                            setattr(self, name, input_param)
                        else:
                            param.copy_(input_param)
                except Exception as ex:
                    action = "swapping" if use_swap_tensors else "copying"
                    error_msgs.append(f'While {action} the parameter named "{key}", '
                                      f'whose dimensions in the model are {param.size()} and '
                                      f'whose dimensions in the checkpoint are {input_param.size()}, '
                                      f'an exception occurred : {ex.args}.'
                                      )
            elif strict:
                missing_keys.append(key)

        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if getattr(self.__class__, "set_extra_state", Module.set_extra_state) is not Module.set_extra_state:
            if extra_state_key in state_dict:
                self.set_extra_state(state_dict[extra_state_key])
            elif strict:
                missing_keys.append(extra_state_key)
        elif strict and (extra_state_key in state_dict):
            unexpected_keys.append(extra_state_key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix) and key != extra_state_key:
                    input_name = key[len(prefix):].split(".", 1)
                    # Must be Module if it have attributes
                    if len(input_name) > 1:
                        if input_name[0] not in self._modules:
                            unexpected_keys.append(key)
                    elif input_name[0] not in local_state:
                        unexpected_keys.append(key)

    # 加载state_dict
    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True, assign: bool = False):
        r"""Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

        If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        .. warning::
            If :attr:`assign` is ``True`` the optimizer must be created after
            the call to :attr:`load_state_dict` unless
            :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
            assign (bool, optional): When ``False``, the properties of the tensors
                in the current module are preserved while when ``True``, the
                properties of the Tensors in the state dict are preserved. The only
                exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s
                for which the value from the module is preserved.
                Default: ``False``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing any keys that are expected
                    by this module but missing from the provided ``state_dict``.
                * **unexpected_keys** is a list of str containing the keys that are not
                    expected by this module but present in the provided ``state_dict``.

        Note:
            If a parameter or buffer is registered as ``None`` and its corresponding key
            exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
            ``RuntimeError``.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = OrderedDict(state_dict)
        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module, local_state_dict, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            if assign:
                local_metadata['assign_to_params_buffers'] = assign
            module._load_from_state_dict(
                local_state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + '.'
                    child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                    load(child, child_state_dict, child_prefix)  # noqa: F821

            # Note that the hook can modify missing_keys and unexpected_keys.
            incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
            for hook in module._load_state_dict_post_hooks.values():
                out = hook(module, incompatible_keys)
                assert out is None, (
                    "Hooks registered with ``register_load_state_dict_post_hook`` are not"
                    "expected to return new values, if incompatible_keys need to be modified,"
                    "it should be done inplace."
                )

        load(self, state_dict)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join(f'"{k}"' for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join(f'"{k}"' for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def _named_members(self, get_members_fn, prefix='', recurse=True, remove_duplicate: bool = True):
        r"""Help yield various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                if remove_duplicate:
                    memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    # 返回模块的parameters迭代器
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        r"""Return an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for param in model.parameters():
            >>>     print(type(param), param.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
            self,
            prefix: str = '',
            recurse: bool = True,
            remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        r"""Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
            remove_duplicate (bool, optional): whether to remove the duplicated
                parameters in the result. Defaults to True.

        Yields:
            (str, Parameter): Tuple containing the name and parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, param in self.named_parameters():
            >>>     if name in ['bias']:
            >>>         print(param.size())

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
        yield from gen

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        r"""Return an iterator over module buffers.

        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            torch.Tensor: module buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for buf in model.buffers():
            >>>     print(type(buf), buf.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[Tuple[str, Tensor]]:
        r"""Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool, optional): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module. Defaults to True.
            remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

        Yields:
            (str, torch.Tensor): Tuple containing the name and buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, buf in self.named_buffers():
            >>>     if name in ['running_var']:
            >>>         print(buf.size())

        """
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
        yield from gen

    # named_children的一个子方法，只返回结构
    def children(self) -> Iterator['Module']:
        r"""Return an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for name, module in self.named_children():
            yield module

    # 返回子模块及其名称
    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        r"""Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

        Yields:
            (str, Module): Tuple containing a name and child module

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)

        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator['Module']:
        r"""Return an iterator over all modules in the network.

        Yields:
            Module: a module in the network

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
            ...     print(idx, '->', m)

            0 -> Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            1 -> Linear(in_features=2, out_features=2, bias=True)

        """
        for _, module in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', remove_duplicate: bool = True):
        r"""Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (str, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
            ...     print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                yield from module.named_modules(memo, submodule_prefix, remove_duplicate)

    # 递归子模块设置为设置为train状态，调用的是tensor的train方法
    def train(self: T, mode: bool = True) -> T:
        r"""这只对某些模块有影响。如果它们受到影响，请参阅特定模块的文档，了解它们在培训/评估模式下的行为细节，
        例如：class:“Dropout”、：class:`BatchNorm`等。
        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    # 同上
    def eval(self: T) -> T:
        r"""Set the module in evaluation mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.eval()` and several similar mechanisms that may be confused with it.

        Returns:
            Module: self
        """
        return self.train(False)

    # 同上遍历
    def requires_grad_(self: T, requires_grad: bool = True) -> T:
        r"""Change if autograd should record operations on parameters in this module.

        This method sets the parameters' :attr:`requires_grad` attributes
        in-place.

        This method is helpful for freezing part of the module for finetuning
        or training parts of a model individually (e.g., GAN training).

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.requires_grad_()` and several similar mechanisms that may be confused with it.

        Args:
            requires_grad (bool): whether autograd should record operations on
                                  parameters in this module. Default: ``True``.

        Returns:
            Module: self
        """
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        r"""Reset gradients of all model parameters.

        See similar function under :class:`torch.optim.Optimizer` for more context.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`torch.optim.Optimizer.zero_grad` for details.
        """
        if getattr(self, '_is_replica', False):
            warnings.warn(
                "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
                "The parameters are copied (in a differentiable manner) from the original module. "
                "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
                "If you need gradients in your forward method, consider using autograd.grad instead.")

        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    # share_memory_ 方法用于将张量移动到共享内存中，这允许多个进程可以访问该张量而无需复制数据。
    # 这个方法是 Tensor 类的一个实例方法，并且可以在 Module 类中通过 _apply 方法间接使用。
    def share_memory(self: T) -> T:
        r"""See :meth:`torch.Tensor.share_memory_`."""
        return self._apply(lambda t: t.share_memory_())

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module.

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ''

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    # 用于返回对象的属性列表。在 PyTorch 的 Module 类中，这个方法被重写，以提供更具体的行为，
    # 尤其是在列出模块的属性时。以下是对 Module 类中 __dir__ 方法的解释：
    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())          
        parameters = list(self._parameters.keys())  
        modules = list(self._modules.keys())        
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers

        # 删除不合法的Python变量名的attr
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    # 用于创建模块的一个副本，这个副本通常用于数据并行操作。
    # 这个方法的目的是为数据并行操作创建一个模块的副本，其中副本不会复制原始模块的参数和缓冲区数据，而是引用原始模块的数据。这在多GPU训练时非常有用，可以减少内存使用并提高效率。
    def _replicate_for_data_parallel(self):
        replica = self.__new__(type(self))
        replica.__dict__ = self.__dict__.copy()      # 复制了原始模块实例的 __dict__ 属性到副本中。__dict__ 包含了模块的所有属性和值。

        # 副本本身没有参数，副本引用原始模块。
        replica._parameters = OrderedDict()          # 初始化了一个空的有序字典，用于存储副本的参数。
        replica._buffers = replica._buffers.copy()   # 复制了原始模块的缓冲区到副本，注意这里可能是一个错误，因为 replica 还没有 _buffers 属性，正确的可能是 replica._buffers = self._buffers.copy()。
        replica._modules = replica._modules.copy()
        replica._is_replica = True  # type: ignore[assignment]

        return replica

    # 编译当前模块（Module）的前向传播（forward）过程，并设置Module的_compiled_call_impl属性
    # 这个方法通过调用 torch.compile 函数来实现编译，torch.compile 是 PyTorch 中的一个函数，用于优化和编译模型的计算图，以加速模型的前向和/或反向传播过程。
    def compile(self, *args, **kwargs):
        """
        Compile this Module's forward using :func:`torch.compile`.

        This Module's `__call__` method is compiled and all arguments are passed as-is
        to :func:`torch.compile`.

        See :func:`torch.compile` for details on the arguments for this function.
        """
        self._compiled_call_impl = torch.compile(self._call_impl, *args, **kwargs)

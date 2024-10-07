"""声明对象的使用方式：对外暴露的可使用的类方法和参数，不具体实现
这个parameter.pyi文件是一个类型提示文件，用于为Python的类型检查器（如mypy）提供类型信息。
类型提示文件通常以.pyi扩展名结尾，它们允许开发者指定函数、方法、类及其成员的预期类型，
从而提高代码的可读性和可维护性，同时帮助静态类型检查器捕获潜在的类型错误。
parameter.py 和 parameter.pyi 文件之间的关系可以理解为实现（implementation）和类型声明（type declaration）的关系。
parameter.py 文件包含了类的实际Python代码实现，而 parameter.pyi 文件提供了这些类的类型注解，用于静态类型检查。
类型注解主要用于提供额外的代码信息，帮助开发者和工具理解代码应该如何使用。
它们不会强制改变Python代码的行为，因为Python是动态类型语言，但它们可以提高代码的可读性和可维护性，以及帮助捕获类型相关的错误。

# mypy: allow-untyped-defs是一个mypy的配置指令，告诉mypy允许在定义中不使用类型注解。
...符号在这里用作类型注解，表示该位置的类型信息未知或未指定。在mypy的上下文中，这可以用于指示函数的返回类型是动态的，或者在某些情况下，表示函数参数的具体值不重要。
"""

# mypy: allow-untyped-defs
import builtins
from typing import Optional, Tuple

import torch
from torch import Tensor

class Parameter(Tensor):
    """
    通常，类型注解文件（.pyi）会包含公共API的类型信息，而 __new__ 方法在很多情况下是内部实现细节，可能不需要在类型注解中体现。
    类型检查器主要关注对象的使用方式，而不是它们的创建方式。
    """
    def __init__(
        self,
        data: Tensor = ...,
        requires_grad: builtins.bool = ...,
    ): ...

def is_lazy(param: Tensor): ...

class UninitializedParameter(Tensor):
    def __init__(
        self,
        data: Tensor = ...,
        requires_grad: builtins.bool = ...,
    ): ...
    def materialize(
        self,
        shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ): ...

class UninitializedBuffer(Tensor):
    def __init__(
        self,
        data: Tensor = ...,
        requires_grad: builtins.bool = ...,
    ): ...
    def materialize(
        self,
        shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ): ...

"""Overview：本文件总行数为5156行，定义了26个函数，8个类（其中6个特定于SQuAD任务），核心是PreTrainedModel类（3390行），是所有预训练模型的基类
（1）1-50行：        原Liscence信息，现为总体信息
（2）51-221行：      import和全局变量设置（16-34标准库导入、36-41/111-133第三方库、43-103本地导入），含两个函数（is_fsdp_enabled、is_local_dist_rank_0）
（3）222-984:        21个函数
    1）no_init_weights：在全局环境中不初始化权重，加速模型加载
    2）——6）5个信息获取函数：获取给定 parameter、weights 的设备、数据类型
            get_parameter_device（获取给定 parameter 的设备）
            get_first_parameter_dtype（获取首个参数的数据类型，无论其是否为浮点数）
            get_first_parameter_dtype（旨在返回模型中第一个浮点类型的 dtype）
            get_state_dict_float_dtype（返回模型权重字典中浮点数的类型）
            get_state_dict_dtype（回模型权重字典中浮点数的类型）
    7）dtype_byte_size：返回指定数据类型
    8）check_support_param_buffer_assignment：检查`model_to_load`是否支持参数缓冲区buffer分配
    9）-11）保存、加载模型的分片检查点（checkpoint）、状态字典
            shard_checkpoint、load_sharded_checkpoint、load_state_dict
    12）set_initialized_submodules：标记模型中的子模块是否已经根据 state_dict 进行初始化
    13）_end_ptr：如果张量是更大张量的切片，计算 PyTorch 张量在内存中所占的最后一个字节的位置（即结束地址）
    14）-16）判断模块权重共享、张量共享底层存储
            _get_tied_weight_keys：递归地获取模块及其子模块中的所有“共享权重”（tied weights）键
            _find_disjoint：识别一组张量中哪些是互不重叠的（disjoint）
            _find_identical：对上述共享相同底层存储的的集合，判断其中哪些是共享相同底层存储且完全相同的（identical），哪些只是共享底层存储但不完全相同的（shared）
    17）-20）模型state_dict加载、转换：如meta设备、正常设备等
        _load_state_dict_into_model：将 PyTorch 的 state_dict 加载到给定的模型中
        find_submodule_and_param_name：根据参数的完整键名（long_key）找到模型中的子模块和参数名
        _move_model_to_meta：在加载模型状态字典（state_dict）之前，将模型中的某些参数或缓冲区移到 meta 设备上。
        _load_state_dict_into_meta_model：将实际占用内存的参数加载到之前实例化的模型类（meta 设备（虚拟，有shape但不占内存））
    21）_add_variant：将一个可选的变量名称 (variant) 插入到字符串权重文件名 (weights_name) 中
（4）985-1293：      class ModuleUtilsMixin，    309行，提供了一些针对`torch.nn.Modules`的实用程序, 用作mixin. 被类混合继承
（5）1294-4684：     class PreTrainedModel，     3390行，核心，所有预训练模型的基类
（6）4685-5097：     6个类，                      412行，，构建和操作基于Transformer架构的问答模型，特别是针对SQuAD（斯坦福问答任务）的模型
    1）4685-4721:      class PoolerStartLogits，   37行，用于计算SQuAD任务中起始位置的对数几率（logits）
    2）4722-4790：     class PoolerEndLogits，     68行，用于计算SQuAD任务中结束位置的对数几率
    3）4791-4854：     class PoolerAnswerClass，   64行，用于SQuAD 2.0任务，预测问题是否有可能的答案。
    4）4855-4881：     class SquadHeadOutput，     30行，使用 dataclass 装饰的类，定义了SQuAD模型输出的数据结构
    5）4882-4998：     class SQuADHead，           116行，SQuAD模型的“头部”，负责整合起始位置、结束位置和答案类别的预测
    6）4999-5097：     class SequenceSummary，     98行，从序列的隐藏状态中生成单一向量摘要。
（7）5098-5156：3个函数
    1）unwrap_model：用于递归地从潜在的容器（如分布式训练中使用的）中解包模型。它确保无论模型被包装了多少层
    2）expand_device_map：用于扩展设备映射，将参数名映射到它们应该被放置的设备上。它处理了模型参数和设备之间的对应关系。
    3）get_disk_only_shard_files）：返回只包含被卸载到磁盘上的权重的分片文件列表。它用于管理和优化模型的权重存储，特别是在模型太大不适合完全加载到内存中时。










"""
import collections                          # 提供高性能的容器数据类型，如namedtuple, deque, Counter, OrderedDict等。用于构建和操作复杂的数据结构。
import copy                                 # 提供了浅拷贝（copy.copy）和深拷贝（copy.deepcopy）功能，可以复制对象，避免对原对象的修改影响到副本。
import functools                            # 提供高阶函数工具，比如partial（用于部分应用函数）、wraps（用于装饰器），可以简化函数式编程和装饰器的使用。
from functools import partial, wraps            # partial：用于"部分应用"函数，固定某些参数的值，创建一个新的函数，该函数具有比原函数更少的参数。这在需要重复调用某个函数，但大部分参数都是相同的场景下非常有用
                                                # wraps：将被装饰函数的元数据（如名称、文档字符串等）拷贝到装饰器中生成的新函数上。这样可以确保装饰后的函数保留原始函数的重要属性。
import gc                                   # 控制Python的垃圾回收器，可以手动执行垃圾回收或调整回收策略。
import importlib.metadata                   # 用于访问和管理 Python 包的元数据。元数据包括包的名称、版本号、依赖项、安装路径等信息。这些信息通常存储在 METADATA 或 PKG-INFO 文件中，位于 Python 包的分发档案中（例如 .dist-info 目录下）
import inspect                              # 用于检查和获取活跃对象的信息，例如函数的参数、源代码、类的层次结构等，主要用于调试和反射。
import itertools                            # 提供高效的迭代器函数，用于构建复杂的迭代结构，常用于处理组合、排列、无限迭代等。
import json                                 
import os                                   # 提供与操作系统交互的功能，如文件路径操作、环境变量获取等，是Python中进行系统操作的基础库
import re                                   
import shutil                               # 提供高级文件操作，如复制、移动、删除文件和目录等，是文件操作的工具库。
import tempfile                             # 用于创建临时文件和目录，通常用于需要临时存储数据的场景，如测试或中间结果保存
import warnings                             # 提供控制Python警告信息的功能
from contextlib import contextmanager       # 一个装饰器，用于创建上下文管理器，简化资源的分配和释放过程
from dataclasses import dataclass           # 简化了类的定义，自动生成初始化、表示和比较方法，常用于数据存储类。
from threading import Thread                # 提供多线程支持，Thread类用于创建和管理线程，实现并发操作
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union       # 提供类型提示工具，帮助定义变量和函数参数的类型，提高代码可读性和静态检查能力
from zipfile import is_zipfile              # 提供ZIP文件处理功能，is_zipfile函数用于判断文件是否为ZIP格式

import torch
from huggingface_hub import split_torch_state_dict_into_shards                  # 将大型模型的状态字典拆分为多个部分，方便存储和分发。
from packaging import version                                   # 用于版本比较的工具，提供符合PEP 440规范的版本解析和比较功能
from torch import Tensor, nn                                    # Tensor是PyTorch中的基本数据结构；nn模块包含了构建神经网络的基本组件。
from torch.nn import CrossEntropyLoss, Identity                 # CrossEntropyLoss是常用的损失函数，适用于分类任务；Identity是一个空操作层，通常用于调试或简化代码。
from torch.utils.checkpoint import checkpoint                   # 用于实现模型的检查点保存，节省内存的同时保持计算图的可用性，适用于训练大模型

from .activations import get_activation                                                     # 从当前包的 activations 模块中导入一个函数，通常用于获取激活函数。激活函数在神经网络中用于非线性变换。
from .configuration_utils import PretrainedConfig                                           # 预训练模型的配置基类。它可能包含模型的超参数、架构信息等。
from .dynamic_module_utils import custom_object_save                                        # 保存自定义对象。它可能与模型的保存或序列化有关。
from .generation import GenerationConfig, GenerationMixin                                   # 用于配置生成模型（如文本生成）的类、混入类为模型添加生成相关的功能（混入类不定义新的类型，而是打包一些方法，以便在其他类中重用）
from .integrations import PeftAdapterMixin, deepspeed_config, is_deepspeed_zero3_enabled    # 集成模型适配器（adapters）的混入类、配置DeepSpeed的工具、检查是否启用了DeepSpeed的ZeRO-3优化技术
from .pytorch_utils import (  # noqa: F401                      # 用于告诉代码检查工具忽略导入但未使用的警告
    Conv1D,                                                     # 1D卷积层的实现
    apply_chunking_to_forward,                                  # 分块处理输入的函数，通常用于节省内存
    find_pruneable_heads_and_indices,                           # 查找可以修剪的注意力头和索引，用于模型压缩。
    id_tensor_storage,                                          # 用于标识张量存储的函数
    is_torch_greater_or_equal_than_1_13,                        # 检查当前PyTorch版本是否大于或等于1.13的函数
    prune_conv1d_layer,                                         # 修剪1D卷积层的函数、通用的层修剪函数、修剪线性层（全连接层）的函数
    prune_layer,
    prune_linear_layer,
)
from .quantizers import AutoHfQuantizer, HfQuantizer            # 自动量化器、通用量化器，用于减少模型的大小或提高推理速度。
from .quantizers.quantizers_utils import get_module_from_name   # 工具函数，用于根据名称获取模块，可能用于动态加载或管理模型组件。
from .safetensors_conversion import auto_conversion             # 自动转换模型权重为safetensors更安全或更高效的格式。
from .utils import (                    # 从本地utils工具箱中导入：蓝色为字符串、绿色为类、棕色为函数
    ACCELERATE_MIN_VERSION,             # 定义了最低支持的Accelerate库版本，            字符串
    ADAPTER_SAFE_WEIGHTS_NAME,          # 适配器的（安全）权重文件名
    ADAPTER_WEIGHTS_NAME, 
    CONFIG_NAME,                        # 配置文件的名称
    DUMMY_INPUTS,                       # 用于测试或占位符的假输入
    FLAX_WEIGHTS_NAME,                  # 框架的权重文件名
    SAFE_WEIGHTS_INDEX_NAME,            # 安全权重索引文件名、安全权重名
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,                   # TensorFlow 1和2 的权重文件名
    TF_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,                 # 权重索引文件名、通用权重文件名
    WEIGHTS_NAME,
    ContextManagers,                    # 上下文管理器集合                              类
    ModelOutput,                        # 模型输出的标准格式。
    PushToHubMixin,                     # 混入类，允许将模型推送到Hugging Face Hub
    cached_file,                        # 用于缓存文件的函数                            函数
    copy_func,                          # 复制函数的工具
    download_url,                       # 下载URL资源的函数
    extract_commit_hash,                # 提取Git提交哈希值的函数
    has_file,                           # 检查文件是否存在的函数
    is_accelerate_available,            # 自动检查各个库是否可用的函数：Accelerate库、BitsAndBytes、Flash Attention 2、是否处于离线模式、Optimum库、Peft库、检查是否为远程URL、Safetensors库、PyTorch SDPA、Torch XLA
    is_bitsandbytes_available,
    is_flash_attn_2_available,
    is_offline_mode,
    is_optimum_available,
    is_peft_available,
    is_remote_url,
    is_safetensors_available,
    is_torch_sdpa_available,
    is_torch_xla_available,
    logging,                            # 日志记录工具
    replace_return_docstrings,          # 替换返回值文档字符串的函数
    strtobool,                          # 将字符串转换为布尔值的函数
)
from .utils.hub import convert_file_size_to_int, create_and_tag_model_card, get_checkpoint_shard_files     # 将文件大小转换为整数的函数、模型的文档化（创建并标记模型卡）、 获取检查点碎片文件的函数用于处理大模型的存储和加载。
from .utils.import_utils import (       # 环境变量中表示“真”的值列表。检查SageMaker模型并行是否启用、是否为Torch FX代理、TorchDynamo编译是否启用
    ENV_VARS_TRUE_VALUES,
    is_sagemaker_mp_enabled,
    is_torch_fx_proxy,
    is_torchdynamo_compiling,
)
from .utils.quantization_config import BitsAndBytesConfig, QuantizationMethod      # 量化配置类，用于设置模型量化的参数；量化方法的定义，可能用于选择不同的量化策略。
 
# 从环境变量中获取两个配置参数，并定义了一个警告信息（发出参数重命名的警告）
XLA_USE_BF16 = os.environ.get("XLA_USE_BF16", "0").upper()                         # 通常用于控制是否在 XLA（Accelerated Linear Algebra，Google 的加速库）中使用 BF16（bfloat16）数据类型。BF16 是一种用于深度学习的16位浮点数格式，可以提高计算效率。
XLA_DOWNCAST_BF16 = os.environ.get("XLA_DOWNCAST_BF16", "0").upper()               # 用于控制是否将数据类型自动转换为 BF16
PARAM_RENAME_WARNING = "A parameter name that contains `{}` will be renamed internally to `{}`. Please use a different name to suppress this warning."

# 如果accelerate加速器库、safetensors模型权重格式可用，再导入一些
if is_accelerate_available():
    from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
    from accelerate.hooks import add_hook_to_module
    from accelerate.utils import (
        check_tied_parameters_on_same_device,
        extract_model_from_parallel,
        find_tied_parameters,
        get_balanced_memory,
        get_max_memory,
        load_offloaded_weights,
        offload_weight,
        save_offload_index,
        set_module_tensor_to_device,
    )

    accelerate_version = version.parse(importlib.metadata.version("accelerate"))
    if accelerate_version >= version.parse("0.31"):
        from accelerate.utils.modeling import get_state_dict_from_offload

if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file

logger = logging.get_logger(__name__)                                                   # 获取一个名为 __name__ 的日志记录器对象


_init_weights = True                                                                    # 用于指示是否初始化权重


def is_fsdp_enabled():                                                                  # 检查是否启用了FSDP（Fully Sharded Data Parallel），这是一个用于分布式训练的大模型并行化策略。
    return (                                                                                # 四个条件都为True才可以：torch是否支持分布式、是否初始化分布式。
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False")) == 1                  # 检查环境变量 ACCELERATE_USE_FSDP 是否被设置为 True（即 "1"）。strtobool 将字符串转换为布尔值。
        and strtobool(os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "False")) == 1       # 同样检查环境变量 FSDP_CPU_RAM_EFFICIENT_LOADING 是否被设置为 True
    )


def is_local_dist_rank_0():                                                             # 检查当前进程是否为本地的分布式 rank 0，即本地的主进程
    return (                                                                                # 检查 PyTorch 的分布式计算是否可用、PyTorch 的分布式计算是否已初始化、检查环境变量 LOCAL_RANK 是否等于 0（本地主进程编号）
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and int(os.environ.get("LOCAL_RANK", -1)) == 0
    )


if is_sagemaker_mp_enabled():                                                           # 如果启用了 Amazon SageMaker 模型并行（SageMaker Model Parallel），导入一些库并设置一个字符串
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")     # 检查 SageMaker 模型并行库的版本是否大于或等于 1.10，并将结果存储在 IS_SAGEMAKER_MP_POST_1_10 中。
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_peft_available():                                                                 # 检查是否可用 参数高效微调库PEFT（Parameter-Efficient Fine-Tuning）库。如果 PEFT 可用，则导入 find_adapter_config_file 函数。
    from .utils import find_adapter_config_file

TORCH_INIT_FUNCTIONS = {                              # 定义了一个字典 TORCH_INIT_FUNCTIONS，将 PyTorch 中用于初始化神经网络权重的各种函数（值value）————字符串形式的键key。下面是多种初始化方法
    "uniform_": nn.init.uniform_,                           # 均匀分布、正态分布、截断正态分布、常数、 Xavier 均匀分布、 Xavier 正态分布、Kaiming 均匀分布、Kaiming 正态分布、均匀分布初始化（非 in-place 版本）、正态分布初始化（非 in-place 版本）
    "normal_": nn.init.normal_,
    "trunc_normal_": nn.init.trunc_normal_,
    "constant_": nn.init.constant_,
    "xavier_uniform_": nn.init.xavier_uniform_,
    "xavier_normal_": nn.init.xavier_normal_,
    "kaiming_uniform_": nn.init.kaiming_uniform_,
    "kaiming_normal_": nn.init.kaiming_normal_,
    "uniform": nn.init.uniform,
    "normal": nn.init.normal,
    "xavier_uniform": nn.init.xavier_uniform,
    "xavier_normal": nn.init.xavier_normal,
    "kaiming_uniform": nn.init.kaiming_uniform,
    "kaiming_normal": nn.init.kaiming_normal,
}

# （1）在全局环境中不初始化权重，加速模型加载
@contextmanager                                                 # 上下文装饰器，用于将一个生成器函数转换为上下文管理器（即支持 with 语句的对象）。定义需要在进入和退出上下文时执行特定操作的函数
def no_init_weights(_enable=True):                              # 也可以自己定义
    """
    上下文管理器，在全局环境中不初始化权重，加速模型加载
    一般用法with ContextManagers(init_contexts): model = cls(config)
    生成器函数是一种特殊类型的函数，它使用 yield 关键字来生成一个值序列，每次调用时都会暂停并返回一个值，而不是一次性返回所有结果。当生成器函数被调用时，它返回的是一个迭代器对象（生成器对象），而不是立即执行整个函数体。
    """
    global _init_weights
    old_init_weights = _init_weights

    if _enable:
        _init_weights = False                                   # 禁用权重初始化。

        def _skip_init(*args, **kwargs):                        # 定义了一个空函数 _skip_init，它不做任何操作，替代 PyTorch 中的初始化函数，使得在调用初始化函数时什么都不会发生。
            pass

        # # Save the original initialization functions          # 遍历字典 TORCH_INIT_FUNCTIONS 中的所有初始化函数，使用 setattr 函数将 torch.nn.init 模块中的所有初始化函数替换为 _skip_init。这样一来，在上下文中调用这些初始化函数时，实际上什么都不会发生，从而跳过了权重初始化。
        for name, init_func in TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, _skip_init)
    try:                                                        # try...finally 结构用于确保无论在 try 块中是否发生异常，finally 块中的代码总是会执行。这对于需要进行资源清理、状态恢复、或者其他需要在代码结束时确保执行的操作特别重要。
        yield                                                   # yield两个作用：将控制权返回给调用者、返回一个值（如果后面跟着的话）。
    finally:                                                    # 将在退出上下文时执行，无论是否发生异常。
        _init_weights = old_init_weights                        # 恢复 _init_weights 的原始状态，以便在上下文结束后，权重初始化的行为恢复正常。
        if _enable:                                             # 在启用了 no_init_weights 的情况下执行恢复操作。再次遍历 TORCH_INIT_FUNCTIONS 字典，这次是为了恢复 PyTorch 中原始的初始化函数。
            # 恢复各个模块中的初始化函数，chatglm的finally中只有_init_weights = old_init_weights，即不恢复初始化
            for name, init_func in TORCH_INIT_FUNCTIONS.items():
                setattr(torch.nn.init, name, init_func)

# （2）获取给定 parameter 的设备（如 GPU 或 CPU）。输入可以是nn.Module、GenerationMixin 或 "ModuleUtilsMixin" 的实例
def get_parameter_device(parameter: Union[nn.Module, GenerationMixin, "ModuleUtilsMixin"]):
    try:
        return next(parameter.parameters()).device                                        
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5
        # # 首先尝试从参数的 parameters 方法中获取第一个参数的设备。如果失败（例如参数中没有可用的张量），则使用备用方法在模块的属性中查找张量并返回其设备。
        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device

# （3）获取首个参数的数据类型，无论其是否为浮点数，逻辑同上
def get_first_parameter_dtype(parameter: Union[nn.Module, GenerationMixin, "ModuleUtilsMixin"]):
    """
    Returns the first parameter dtype (can be non-floating) or asserts if none were found.
    """
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch > 1.5

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype

# （4）旨在返回模型中第一个浮点类型的 dtype。如果模型中没有浮点数类型的参数，它将返回找到的最后一个参数的 dtype，或者在一些特定情况下返回一个强制转换后的 dtype。
def get_parameter_dtype(parameter: Union[nn.Module, GenerationMixin, "ModuleUtilsMixin"]):
    """
    Returns the first found floating dtype in parameters if there is one, otherwise returns the last dtype it found.
    """
    last_dtype = None
    for t in parameter.parameters():
        last_dtype = t.dtype
        if t.is_floating_point():
            # Adding fix for https://github.com/pytorch/xla/issues/4152
            # Fixes issue where the model code passes a value that is out of range for XLA_USE_BF16=1
            # and XLA_DOWNCAST_BF16=1 so the conversion would cast it to -inf
            # NOTE: `is_torch_xla_available()` is checked last as it induces a graph break in torch dynamo
            if XLA_USE_BF16 in ENV_VARS_TRUE_VALUES and is_torch_xla_available():
                return torch.bfloat16
            if XLA_DOWNCAST_BF16 in ENV_VARS_TRUE_VALUES and is_torch_xla_available():
                if t.dtype == torch.float:
                    return torch.bfloat16
                if t.dtype == torch.double:
                    return torch.float32
            return t.dtype

    if last_dtype is not None:
        # if no floating dtype was found return whatever the first dtype is
        return last_dtype

    # For nn.DataParallel compatibility in PyTorch > 1.5
    def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
        tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
        return tuples

    gen = parameter._named_members(get_members_fn=find_tensor_attributes)
    last_tuple = None
    for tuple in gen:
        last_tuple = tuple
        if tuple[1].is_floating_point():
            return tuple[1].dtype

    if last_tuple is not None:
        # fallback to the last dtype
        return last_tuple[1].dtype

    # fallback to buffer dtype
    for t in parameter.buffers():
        last_dtype = t.dtype
        if t.is_floating_point():
            return t.dtype
    return last_dtype

# （5）返回模型权重字典中浮点数的类型，如果不存在浮点数则报错
def get_state_dict_float_dtype(state_dict):
    """
    Returns the first found floating dtype in `state_dict` or asserts if none were found.
    """
    for t in state_dict.values():
        if t.is_floating_point():
            return t.dtype

    raise ValueError("couldn't find any floating point dtypes in state_dict")

# （6）返回模型权重字典中浮点数的类型，如果不存在浮点数则返回最后一个数据的类型
def get_state_dict_dtype(state_dict):
    """
    Returns the first found floating dtype in `state_dict` if there is one, otherwise returns the first dtype.
    """
    for t in state_dict.values():
        if t.is_floating_point():
            return t.dtype

    # if no floating dtype was found return whatever the first dtype is
    else:
        return next(state_dict.values()).dtype

# （7）返回指定数据类型（dtype）在内存中占用的字节数
def dtype_byte_size(dtype):
    """
    接受一个 dtype 参数，返回该数据类型占用的字节数。通过正则表达式来解析 dtype 的比特大小，并根据比特数计算字节数。
    布尔类型（torch.bool）是一个特殊情况。布尔类型的值只占用1位（bit），而不是字节。因此，1 / 8 表示布尔类型的大小是1位，转换为字节则是 0.125 字节（1/8 字节）。
    Example:

    ```py
    >>> dtype_byte_size(torch.float32)
    4
    ```
    """
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)_?", str(dtype))             # 使用正则表达式来提取 dtype 中的比特数。[^\d]：匹配一个非数字字符。(\d+)：这是一个捕获组，匹配一个或多个连续的数字。_?：这是一个可选的下划线，匹配零个或一个下划线字符。
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.") # 如果正则表达式未能匹配到比特数（即 bit_search 为 None），则抛出一个 ValueError 异常，提示 dtype 无效。这可能是由于传入了一个无法识别的数据类型。
    bit_size = int(bit_search.groups()[0])                          # 如果正则表达式匹配成功，从 bit_search 中提取第一个捕获组（即比特数），并将其转换为整数。
    return bit_size // 8

# （8）检查`model_to_load`是否支持参数缓冲区buffer分配
def check_support_param_buffer_assignment(model_to_load, state_dict, start_prefix=""):
    """
    model_to_load：待加载的模型。state_dict：包含模型参数的状态字典。start_prefix：用于匹配 state_dict 中键的前缀。
    检查`model_to_load`是否支持参数缓冲区buffer分配（例如在加载空权重时），首先检查模型是否明确禁用它，然后确保状态字典键是模型参数的子集。
    注意：如果我们使用“deepspeed”，我们将完全禁用此功能`
    """
    # 步骤 1: 检查 state_dict 中的键,通过列表推导式检查 state_dict 中是否有任何键以 start_prefix 开头。# 如果找不到任何匹配的键，则函数返回 False，表示无法进行参数缓冲区的赋值。
    if len([key for key in state_dict if key.startswith(start_prefix)]) == 0:    
        return False                                                             

    # 步骤 2: 检查是否启用了 DeepSpeed Zero3，果启用了 DeepSpeed Zero3，函数直接返回 False，因为在这种情况下，通常不支持参数缓冲区赋值。
    if is_deepspeed_zero3_enabled():                                             
        return False

    # 步骤 3: 检查模型是否显式不支持参数缓冲区赋值，检查模型对象 model_to_load 是否有一个名为 _supports_param_buffer_assignment 的属性。
    if not getattr(model_to_load, "_supports_param_buffer_assignment", True):    
        logger.debug(                                  # 如果该属性存在且值为 False，则函数返回 False，并且记录调试日志，说明该模型不支持参数缓冲区赋值，加载速度会因此变慢。                          
            f"{model_to_load.__class__.__name__} does not support param buffer assignment, loading will be slower"
        )
        return False

    # 步骤 4: 检查数据类型是否匹配，获取模型状态字典的第一个键 first_key。检查 state_dict 中是否存在与 start_prefix 加上 first_key 相匹配的键。
    first_key = list(model_to_load.state_dict().keys())[0]
    if start_prefix + first_key in state_dict:                                   
        return state_dict[start_prefix + first_key].dtype == model_to_load.state_dict()[first_key].dtype

    # 步骤 5: 处理特殊情况，如果上述条件都不满足，并且在 state_dict 中找不到与 model_to_load 对应的实际权重（比如在某些测试场景中），函数返回 False。
    return False                                                                 

# （9）保存模型的分片检查点（checkpoint）
def shard_checkpoint(
    state_dict: Dict[str, torch.Tensor], max_shard_size: Union[int, str] = "10GB", weights_name: str = WEIGHTS_NAME
):
    """
    拆分模型状态字典为多个子检查点中，使每个子检查点的最终大小不超过给定大小。
    子检查点是通过按键的顺序迭代`state_dict`来确定的，因此没有进行优化来使每个子检查点尽可能接近通过的最大大小。例如，如果限制为10GB，并且我们有大小为[6GB、6GB、2GB、6GB，2GB、2GB]的权重，则它们将被分片为[6GB]、[6+2GB]、[6+2+2GB]，而不是[6+2+2GB]、[6]、[6GB]。
    <提示警告=｛true｝>
    如果模型的某个权重大于“max_shard_size”，则它最终将位于其自己的子检查点中，该子检查点将具有大于“max_Shared_size”的大小。

    请注意，shard_checkpoint已被弃用，并将在v4.44中删除。我们建议您使用huggingface_hub库中的split_torch_state_dict_into_shards
    </Tip>

    Args:
        state_dict (`Dict[str, torch.Tensor]`): 一个包含模型参数的字典，键是参数名称，值是 torch.Tensor
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            定义单个分片的最大大小，可以是整数（字节数）或字符串（例如 "10GB"）。
            (like `"5MB"`).
        weights_name (`str`, *optional*, defaults to `"pytorch_model.bin"`):
            生成的权重文件的名称。
    """
    logger.warning(
        "Note that `shard_checkpoint` is deprecated and will be removed in v4.44. We recommend you using "
        "split_torch_state_dict_into_shards from huggingface_hub library"
    )
    max_shard_size = convert_file_size_to_int(max_shard_size)         # 将 max_shard_size 从字符串（如 "10GB"）转换为整数（字节数）。
    # 初始化变量
    sharded_state_dicts = [{}]                                        # 用于存储分片的 state_dict。
    last_block_size = 0                                               # 当前分片的大小。
    total_size = 0                                                    # 所有分片的总大小。
    storage_id_to_block = {}                                          # 用于跟踪张量存储的位置，以避免重复存储。

    for key, weight in state_dict.items():                            # 遍历 state_dict 中的每个键值对。
        # 当使用bnb序列化时，状态字典中的权重可以是字符串
        # check: https://github.com/huggingface/transformers/pull/24416 for more details
        if isinstance(weight, str):                                   # 如果权重是字符串，跳过此项（适用于特定序列化方式）。
            continue
        else:
            storage_id = id_tensor_storage(weight)

        # 如果 weight 和另一个张量共享同一存储，并且 weight 设备不是 "meta"，则将其放入同一个分片中。
        if storage_id in storage_id_to_block and weight.device != torch.device("meta"):
            block_id = storage_id_to_block[storage_id]
            sharded_state_dicts[block_id][key] = weight
            continue

        weight_size = weight.numel() * dtype_byte_size(weight.dtype)  
        # 计算权重大小并检查是否需要新建分片
        # 如果这个权重将超过最大大小，我们会进行拆分，但前提是我们在当前分片中至少放置了一个权重。
        if last_block_size + weight_size > max_shard_size and len(sharded_state_dicts[-1]) > 0:
            sharded_state_dicts.append({})
            last_block_size = 0
        
        sharded_state_dicts[-1][key] = weight                          # 添加权重到当前分片
        last_block_size += weight_size                                 # 更新当前分片的大小和总大小。记录 storage_id 与分片索引的映射。
        total_size += weight_size
        storage_id_to_block[storage_id] = len(sharded_state_dicts) - 1

    # 如果只有一个分片，直接返回这个分片，并且不生成索引。
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # 否则，构建索引
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):                   # 为每个分片生成文件名，并构建权重到文件名的映射 weight_map。将每个分片存储在 shards 字典中
        shard_file = weights_name.replace(".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin")
        shard_file = shard_file.replace(
            ".safetensors", f"-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.safetensors"
        )
        shards[shard_file] = shard
        for key in shard.keys():
            weight_map[key] = shard_file

    # 添加元数据并返回结果
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}
    return shards, index

# （10）加载模型的分片检查点（checkpoint）：检查检查点和索引表
def load_sharded_checkpoint(model, folder, strict=True, prefer_safe=True):
    """
    和torch.nn.Module.load_state_dict一样，但是为了加载sharded checkpoint
    [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)
    这种加载是高效执行的：每个检查点分片在RAM中逐一加载，并在加载到模型中后删除。
    
    Args:
        model (`torch.nn.Module`): .                           加载检查点的模型
        folder (`str` or `os.PathLike`): .                     包含分片检查点的文件夹的路径
        strict (`bool`, *optional`, defaults to `True`):       是否严格执行模型状态字典中的键与分片检查点中的键匹配。
        prefer_safe (`bool`, *optional*, defaults to `False`)  如果检查点中同时存在safetensor和PyTorch保存文件，并且“prefer_safe”为True，则将加载safetensor文件。否则，总是尽可能加载pt文件。

    Returns:
        `NamedTuple`: 带有“missing_keys”和“unsected_keys”字段的命名元组
            - `missing_keys` 是一个包含缺失键的str列表
            - `unexpected_keys` 是一个包含意外键的str列表
    """
    # 加载索引：定义了检查点索引文件的路径：
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)

    # 检查标准索引文件和 safetensors 索引文件是否存在。
    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)

    # 如果两个索引文件都不存在，或者 safetensors 文件存在但 safetensors 库不可用，抛出异常。
    if not index_present and not (safe_index_present and is_safetensors_available()):
        filenames = (
            (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME) if is_safetensors_available() else (WEIGHTS_INDEX_NAME,)
        )
        raise ValueError(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {folder}.")
    
    # 决定使用哪种格式的索引文件
    load_safe = False
    if safe_index_present:
        if prefer_safe:
            if is_safetensors_available():              # 如果 safe_index_file 存在，并且 prefer_safe 为 True，且 safetensors 库可用，则选择加载 safetensors 格式。
                load_safe = True
            else:                                       # 如果 safetensors 库不可用，记录警告日志。
                logger.warning(
                    f"Cannot load sharded checkpoint at {folder} safely since safetensors is not installed!"
                )
        elif not index_present:
            load_safe = True                            # 如果标准索引文件不存在，但 safe_index_file 存在，则强制使用 safetensors 格式。

    load_index = safe_index_file if load_safe else index_file

    with open(load_index, "r", encoding="utf-8") as f:  # 打开并加载选定的索引文件（标准或 safetensors）为 index 字典。
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))                                   # 从索引文件中提取所有分片文件的路径，并去重。

    # 检查模型的状态字典与索引文件是否匹配:如果 strict 为 True，并且发现缺失键或多余键，构建错误信息并抛出异常。
    loaded_keys = index["weight_map"].keys()
    model_keys = model.state_dict().keys()
    missing_keys = [key for key in model_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)
    
    # 根据 torch 版本决定是否使用 weights_only 参数。根据是否使用 safetensors 决定加载文件的方法（safe_load_file 或 torch.load）。
    weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
    loader = safe_load_file if load_safe else partial(torch.load, map_location="cpu", **weights_only_kwarg)

    # 加载每个分片并更新模型的状态，使用 strict=False 来允许加载过程中模型状态和检查点状态的部分不匹配。在每次加载后释放内存。
    for shard_file in shard_files:
        state_dict = loader(os.path.join(folder, shard_file))
        model.load_state_dict(state_dict, strict=False)

        # Make sure memory is freed before we load the next state dict.
        del state_dict
        gc.collect()

    # 返回一个包含缺失键和多余键的对象，以便进一步处理或调试。
    return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)

# （11）加载模型状态字典
def load_state_dict(checkpoint_file: Union[str, os.PathLike], is_quantized: bool = False):
    """
    读取PyTorch检查点文件，如果出现格式正确的错误，则返回错误。
    """
    if checkpoint_file.endswith(".safetensors") and is_safetensors_available():
        # 处理 Safetensors 格式：如果是 safetensors 格式，使用 safe_open 打开文件并检查元数据。如果元数据中不包含有效的格式（"pt", "tf", "flax", "mlx"），则抛出错误。如果元数据正确，则使用 safe_load_file 加载文件并返回。
        with safe_open(checkpoint_file, framework="pt") as f:
            metadata = f.metadata()
        if metadata.get("format") not in ["pt", "tf", "flax", "mlx"]:
            raise OSError(
                f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                "you save your model with the `save_pretrained` method."
            )
        return safe_load_file(checkpoint_file)                              
    try:    
        # 处理非 Safetensors 文件，根据 is_deepspeed_zero3_enabled 和 is_fsdp_enabled 的状态，决定是否将 map_location 设置为 "meta"（此时不实际加载数据）或 "cpu"。
        if (
            (is_deepspeed_zero3_enabled() and torch.distributed.is_initialized() and torch.distributed.get_rank() > 0)
            or (is_fsdp_enabled() and not is_local_dist_rank_0())
        ) and not is_quantized:
            map_location = "meta"
        else:
            map_location = "cpu"
        extra_args = {}
        # mmap can only be used with files serialized with zipfile-based format.
        if (
            isinstance(checkpoint_file, str)
            and map_location != "meta"
            and version.parse(torch.__version__) >= version.parse("2.1.0")
            and is_zipfile(checkpoint_file)
        ):
            extra_args = {"mmap": True}
        weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
        return torch.load(
            checkpoint_file,
            map_location=map_location,
            **weights_only_kwarg,
            **extra_args,
        )
    except Exception as e:   
        # 处理加载失败的情况
        try:                 
            # 尝试打开检查点文件并读取前 7 个字符，如果字符为 "version"，则可能是用户克隆了没有安装 git-lfs 的仓库，提示安装 git-lfs 并重新拉取数据。如果不是 version，则抛出 ValueError，提示无法找到文件。
            with open(checkpoint_file) as f:
                if f.read(7) == "version":
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:       
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            # 如果文件不是 UTF-8 编码或其他错误，抛出 OSError，提示无法加载权重，并告知如果从 TensorFlow 2.0 的检查点加载，需要设置 from_tf=True。
            raise OSError(
                f"Unable to load weights from pytorch checkpoint file for '{checkpoint_file}' "
                f"at '{checkpoint_file}'. "
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True."
            )

# （12）标记模型中的子模块是否已经根据 state_dict 进行初始化，并返回未初始化的子模块。
def set_initialized_submodules(model, state_dict_keys):
    """
    遍历模型的所有子模块，找出与当前子模块相关的 state_dict 键。检查 loaded_keys 是否包含当前子模块的所有参数键。如果包含，说明该子模块已初始化。否则，将该子模块添加到 not_initialized_submodules 字典中，
    """
    not_initialized_submodules = {}
    for module_name, module in model.named_modules():
        loaded_keys = {k.replace(f"{module_name}.", "") for k in state_dict_keys if k.startswith(f"{module_name}.")}
        if loaded_keys.issuperset(module.state_dict()):
            module._is_hf_initialized = True
        else:
            not_initialized_submodules[module_name] = module
    return not_initialized_submodules

# （13）如果张量是更大张量的切片，计算 PyTorch 张量在内存中所占的最后一个字节的位置（即结束地址）
def _end_ptr(tensor: torch.Tensor) -> int:
    if tensor.nelement():      #  返回张量中的元素总数。如果张量包含元素（即 nelement() 返回值大于零），则继续计算张量的结束地址：
        stop = tensor.view(-1)[-1].data_ptr() + tensor.element_size()  # 将张量展平为一维，获取展平后张量的最后一个元素，获取其起始内存地址和每个元素占用的字节数。将最后一个元素的起始内存地址与元素的字节大小相加，得到张量的结束地址，即 stop。
    else:                      # 如果张量为空（即没有元素），则直接获取其数据指针。
        stop = tensor.data_ptr()
    return stop

# （14）递归地获取模块及其子模块中的所有“共享权重”（tied weights）键。
def _get_tied_weight_keys(module: nn.Module, prefix=""):
    tied_weight_keys = []
    if getattr(module, "_tied_weights_keys", None) is not None:                                 # 检查 _tied_weights_keys 属性
        names = [f"{prefix}.{k}" if prefix else k for k in module._tied_weights_keys]               # 如果存在该属性且不为 None，则获取其值并将每个键名加上前缀 prefix。如果前缀存在（即 prefix 不为空），则将前缀和键名通过 . 连接起来。
        tied_weight_keys.extend(names)                                                              # 将处理后的键名添加到 tied_weight_keys 列表中。
    if getattr(module, "_dynamic_tied_weights_keys", None) is not None:                         # 检查 _dynamic_tied_weights_keys 属性
        names = [f"{prefix}.{k}" if prefix else k for k in module._dynamic_tied_weights_keys]
        tied_weight_keys.extend(names)
    for name, submodule in module.named_children():                                             # 递归处理子模块：
        local_prefix = f"{prefix}.{name}" if prefix else name                                       # 对每个子模块，构建一个新的前缀 local_prefix，它是当前模块的前缀加上子模块的名称。
        tied_weight_keys.extend(_get_tied_weight_keys(submodule, prefix=local_prefix))              # 递归调用 _get_tied_weight_keys 函数处理子模块，将返回的键名添加到 tied_weight_keys 列表中。
    return tied_weight_keys                                                                     # 返回共享权重键的列表

# （15）识别一组张量中哪些是互不重叠的（disjoint），哪些是共享相同底层存储的（shared）。
def _find_disjoint(tensors: List[Set[str]], state_dict: Dict[str, torch.Tensor]) -> Tuple[List[Set[str]], List[str]]:
    """
    参数说明：
        tensors：一个包含张量名称集合的列表，每个集合中的张量名称可能共享相同的底层存储。
        state_dict：一个字典，键是张量的名称，值是对应的 PyTorch 张量（torch.Tensor）。
    返回值：返回一个元组，包含两个列表：
        shared_tensors：存储共享底层存储的张量名称集合的列表。
        disjoint_tensors：存储不共享底层存储的张量名称列表。
    """
    filtered_tensors = []
    for shared in tensors:                      # 遍历 tensors 列表中的每个集合 shared，
        if len(shared) < 2:                             # 如果集合 shared 中的张量少于 2 个（即没有共享的可能性），直接将其添加到 filtered_tensors，并跳过后续处理。
            filtered_tensors.append(shared)
            continue

        areas = []
        for name in shared:                             # 对于 shared 集合中的每个张量名称 name：获取对应的张量 tensor。使用 tensor.data_ptr() 获取张量的起始内存地址，使用 _end_ptr(tensor) 获取张量的结束内存地址。
            tensor = state_dict[name]
            areas.append((tensor.data_ptr(), _end_ptr(tensor), name))
        areas.sort()                                    # 将起始地址、结束地址和张量名称作为元组添加到 areas 列表中。最后，按起始地址对 areas 列表进行排序。
        # 过滤不重叠的张量
        _, last_stop, last_name = areas[0]
        filtered_tensors.append({last_name})
        for start, stop, name in areas[1:]:
            if start >= last_stop:
                filtered_tensors.append({name})
            else:
                filtered_tensors[-1].add(name)
            last_stop = stop
    # 分类共享和不共享的张量
    disjoint_tensors = []
    shared_tensors = []
    for tensors in filtered_tensors:                    # 遍历 filtered_tensors 列表中的每个集合，
        if len(tensors) == 1:
            disjoint_tensors.append(tensors.pop())          # 如果集合中只有一个张量名称，说明该张量是独立的，将其添加到 disjoint_tensors。
        else:
            shared_tensors.append(tensors)                  # 如果集合中有多个张量名称，说明它们共享底层存储，将集合添加到 shared_tensors。
    return shared_tensors, disjoint_tensors

# （16）对上述共享相同底层存储的的集合，判断其中哪些是共享相同底层存储且完全相同的（identical），哪些只是共享底层存储但不完全相同的（shared）
def _find_identical(tensors: List[Set[str]], state_dict: Dict[str, torch.Tensor]) -> Tuple[List[Set[str]], Set[str]]:
    shared_tensors = []
    identical = []
    for shared in tensors:                          # 遍历 tensors 列表中的每个集合 shared。
        if len(shared) < 2:
            continue

        areas = collections.defaultdict(set)
        for name in shared:                         # 对于 shared 集合中的每个张量名称 name，计算张量的内存区域信息，并将这些信息作为元组 area。将张量名称添加到 areas 字典中，以 area 作为键。
            tensor = state_dict[name]
            area = (tensor.device, tensor.data_ptr(), _end_ptr(tensor))
            areas[area].add(name)
        if len(areas) == 1:                         # 如果 areas 字典中只有一个键，说明集合中的所有张量共享完全相同的内存区域，即这些张量是完全相同的。
            identical.append(shared)
        else:
            shared_tensors.append(shared)           # 如果 areas 字典中有多个键，说明集合中的张量不完全相同，但仍然共享底层存储。
    return shared_tensors, identical

# （17）将 PyTorch 的 state_dict 加载到给定的模型中，包括处理一些旧的参数名格式和递归地加载所有子模块的状态。
def _load_state_dict_into_model(model_to_load, state_dict, start_prefix, assign_to_params_buffers=False):
    # 处理旧格式的键名：
    old_keys = []
    new_keys = []
    for key in state_dict.keys():                               # 遍历 state_dict 中的所有键，如果键中包含 "gamma"，将其替换为 "weight"，并记录旧键和新键；如果键中包含 "beta"，将其替换为 "bias"，并记录旧键和新键。
        new_key = None
        if "gamma" in key:
            logger.warning(PARAM_RENAME_WARNING.format("gamma", "weight"))
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            logger.warning(PARAM_RENAME_WARNING.format("beta", "bias"))
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):            # 更新 state_dict 中的键，将旧键的值移动到新键上，并删除旧键。
        state_dict[new_key] = state_dict.pop(old_key)

    # 复制 state_dict 以便修改
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
        
    # 初始化错误消息列表
    error_msgs = []

    # PyTorch的`_load_from_state_dict`不会复制模块子代中的参数，因此我们需要递归应用该函数。
    # 定义一个内部函数 load 用于递归地将状态字典加载到模型和子模块中。
    def load(module: nn.Module, state_dict, prefix="", assign_to_params_buffers=False):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        local_metadata["assign_to_params_buffers"] = assign_to_params_buffers

        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # 模块和子模块的参数将以前缀开头。如果此状态中没有，我们可以提前退出_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if is_deepspeed_zero3_enabled():
                import deepspeed

                # 在分片模型中，每个分片只有完整state_dict的一部分，因此只收集当前state_dict中的参数。
                named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
                if len(params_to_gather) > 0:
                    # 因为zero3将占位符放在模型参数中，所以此上下文管理器收集（未分区）当前层的参数，然后从状态字典加载，然后再次重新分区
                    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".", assign_to_params_buffers)
    # 调用 load 函数，将状态字典加载到 model_to_load 模型中，使用指定的前缀和 assign_to_params_buffers 标志。
    load(model_to_load, state_dict, prefix=start_prefix, assign_to_params_buffers=assign_to_params_buffers)
    # 删除`state_dict`，以便GC可以更早地收集它。请注意，`state_dict`是参数的副本，因此删除它是安全的。
    del state_dict

    return error_msgs

# （18）根据参数的完整键名（long_key）找到模型中的子模块和参数名。
def find_submodule_and_param_name(model, long_key, start_prefix):
    """
    辅助工具util，用于查找最后一个子模块和参数/缓冲区名称。如果提供了`start_prefix`，它将从键的开头删除
    """
    if len(start_prefix) > 0 and long_key.startswith(start_prefix):    # 如果 start_prefix 非空且 long_key 以该前缀开头，将该前缀去掉。
        long_key = ".".join(long_key.split(".")[1:])                   # 具体操作是将 long_key 按点 . 分隔为列表，去掉第一个元素（即前缀），然后再用点 . 拼接回字符串。

    split_key = long_key.split(".")                                    # 初始化和分割键名：将 long_key 按照点 . 分割成列表 split_key，其中每个元素都是模型中一个子模块或参数的名称。将 submodule 初始化为 model，准备逐步查找子模块。
    submodule = model
    while len(split_key) > 1:                                          # 使用 while 循环遍历 split_key，查找模型中的子模块：如果当前 submodule 中存在名为 split_key[0] 的属性，将 submodule 更新为该属性（即找到的子模块），并删除 split_key 中的第一个元素。
        if hasattr(submodule, split_key[0]):
            submodule = getattr(submodule, split_key[0])
            del split_key[0]
        else:
            submodule = None
            break
    if submodule == model:
        submodule = None
    return submodule, split_key[0]

# （19）在加载模型状态字典（state_dict）之前，将模型中的某些参数或缓冲区移到 meta 设备上。这种方法通常用于处理大型模型，确保在内存不足的情况下仍然可以加载它们。
def _move_model_to_meta(model, loaded_state_dict_keys, start_prefix):
    """
    将模型中的`loaded_state_dict_keys`移动到元设备，从而释放这些参数占用的内存。
    `start_prefix用于将名称插入模型键中的模型，例如“bert.pooler.dense.weight”中的“bert”`
    """
    for k in loaded_state_dict_keys:                            # 遍历 loaded_state_dict_keys 列表中的每个键 k。使用 find_submodule_and_param_name 函数根据键名 k 和前缀 start_prefix 找到模型中的子模块 submodule 及其对应的参数名 param_name。
        submodule, param_name = find_submodule_and_param_name(model, k, start_prefix)
        if submodule is not None:
            # 选如果 submodule 不为 None，则获取子模块中参数名为 param_name 的属性值 new_val。
            new_val = getattr(submodule, param_name)
            if isinstance(new_val, torch.nn.Parameter):
                # 如果 new_val 是一个 torch.nn.Parameter 对象，则将其转换为 meta 设备上的新 Parameter 对象。
                new_val = torch.nn.Parameter(new_val.to("meta"))
            else:
                # 否则，将 new_val 转移到 meta 设备。
                new_val = new_val.to("meta")
            setattr(submodule, param_name, new_val)

# （20）将实际占用内存的参数加载到之前实例化的模型类（meta 设备（虚拟，有shape但不占内存））
def _load_state_dict_into_meta_model(
    model,                          # 待加载的模型对象。
    state_dict,                     # 包含模型参数的状态字典。    
    loaded_state_dict_keys,         # 已加载的状态字典的键列表。暂时保留，但可以删除，见下文
    start_prefix,                   # 用于处理键名前缀。
    expected_keys,
    device_map=None,    
    offload_folder=None,            # 用于将参数卸载到磁盘的文件夹。
    offload_index=None,             # 用于跟踪卸载参数的索引。
    state_dict_folder=None,         # 状态字典文件夹。
    state_dict_index=None,
    dtype=None,
    hf_quantizer=None,              # 用于量化的对象。
    is_safetensors=False,
    keep_in_fp32_modules=None,      # 指定哪些模块的参数应保持为 FP32。
    unexpected_keys=None,           # 意外的状态字典键，传递“unexpected”以从量化项中清除
):
    """
    这有点类似于`_load_state_dict_into_model`，但处理的是在`meta`设备上具有部分或全部参数的模型。它将模型参数替换为“state_dict”中的数据，同时将参数移回正常设备，但仅限于“loaded_state_dict_keys”。
    在初始化阶段，模型的参数可能被放置在 meta 设备上。这意味着模型知道参数的形状和类型，但这些参数还没有实际占用内存。
    核心任务是将 state_dict 中的实际参数从磁盘加载到模型中，并将这些参数从 meta 设备移动到指定的实际设备（如 GPU 或 CPU）。
    `start_prefix` is used for models which insert their name into model keys, e.g. `bert` in
    `bert.pooler.dense.weight`
    """
    #XXX：要实现的其余功能与_load_state_dict_into_model完全兼容
    #-deepspeed zero 3 support
    #-需要复制元数据（如果有的话）-请参阅_load_state_dict_into_model
    #-处理error_msgs-模仿模块中的错误处理_load_from_state_dict（）
    #-是否存在某些keys不在`loaded_state_dict_keys`中的情况，在这种情况下，它们将不会被加载。

    # 警告和键名替换：处理旧格式键名的替换，比如将 gamma 替换为 weight，beta 替换为 bias。
    error_msgs = []
    old_keys = []
    new_keys = []
    is_quantized = hf_quantizer is not None
    for key in state_dict.keys():
        new_key = None
        if "gamma" in key:
            logger.warning(PARAM_RENAME_WARNING.format("gamma", "weight"))
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            logger.warning(PARAM_RENAME_WARNING.format("beta", "bias"))
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
    
    # 处理 state_dict 中的每个参数：检查参数是否在已加载的键中，并且是否是预期的键。如果键名以 start_prefix 开头，则去掉前缀
    is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")

    for param_name, param in state_dict.items():
        # First part of the test is always true as load_state_dict_keys always contains state_dict keys.
        if param_name not in loaded_state_dict_keys or param_name not in expected_keys:
            continue

        if param_name.startswith(start_prefix):
            param_name = param_name[len(start_prefix) :]

        module_name = param_name
        set_module_kwargs = {}

        # 处理数据类型转换：将浮点数据类型转换为传递的“dtype”，但float8_e4m3fn类型除外。我们还希望将缓冲区/参数保存在int/uint/bool中，而不是强制转换它们
        is_param_float8_e4m3fn = is_torch_e4m3fn_available and param.dtype == torch.float8_e4m3fn
        if dtype is not None and torch.is_floating_point(param) and not is_param_float8_e4m3fn:
            if (
                keep_in_fp32_modules is not None
                and any(
                    module_to_keep_in_fp32 in param_name.split(".") for module_to_keep_in_fp32 in keep_in_fp32_modules
                )
                and dtype == torch.float16
            ):
                param = param.to(torch.float32)

                # For backward compatibility with older versions of `accelerate`
                # TODO: @sgugger replace this check with version check at the next `accelerate` release
                if "dtype" in list(inspect.signature(set_module_tensor_to_device).parameters):
                    set_module_kwargs["dtype"] = torch.float32
            else:
                param = param.to(dtype)

        # 为了与PyTorch load_state_dict兼容，它将状态字典数据类型转换为模型中的现有数据类型，并使用`param.copy_（input_param）`来保持模型中参数的连续性。
        # Reference: https://github.com/pytorch/pytorch/blob/db79ceb110f6646523019a59bbd7b838f43d4a86/torch/nn/modules/module.py#L2040C29-L2040C29
        # 兼容性检查和设备映射
        old_param = model
        splits = param_name.split(".")
        for split in splits:
            old_param = getattr(old_param, split)
            if old_param is None:
                break
        if old_param is not None:
            if dtype is None:
                param = param.to(old_param.dtype)

            if old_param.is_contiguous():
                param = param.contiguous()

        set_module_kwargs["value"] = param

        if device_map is None:
            param_device = "cpu"
        else:
            # find next higher level module that is defined in device_map:
            # bert.lm_head.weight -> bert.lm_head -> bert -> ''
            while len(module_name) > 0 and module_name not in device_map:
                module_name = ".".join(module_name.split(".")[:-1])
            if module_name == "" and "" not in device_map:
                # TODO: group all errors and raise at the end.
                raise ValueError(f"{param_name} doesn't have any device set.")
            param_device = device_map[module_name]

        # 参数卸载和加载：根据 param_device 执行参数卸载或加载操作。如果使用量化，则使用量化器处理参数。
        if param_device == "disk":
            if not is_safetensors:
                offload_index = offload_weight(param, param_name, offload_folder, offload_index)
        elif param_device == "cpu" and state_dict_index is not None:
            state_dict_index = offload_weight(param, param_name, state_dict_folder, state_dict_index)
        elif (
            not is_quantized
            or (not hf_quantizer.requires_parameters_quantization)
            or (
                not hf_quantizer.check_quantized_param(
                    model, param, param_name, state_dict, param_device=param_device, device_map=device_map
                )
            )
        ):
            # For backward compatibility with older versions of `accelerate` and for non-quantized params
            set_module_tensor_to_device(model, param_name, param_device, **set_module_kwargs)
        else:
            hf_quantizer.create_quantized_param(model, param, param_name, param_device, state_dict, unexpected_keys)
            # For quantized modules with FSDP/DeepSpeed Stage 3, we need to quantize the parameter on the GPU
            # and then cast it to CPU to avoid excessive memory usage on each GPU
            # in comparison to the sharded model across GPUs.
            if is_fsdp_enabled() or is_deepspeed_zero3_enabled():
                module, tensor_name = get_module_from_name(model, param_name)
                value = getattr(module, tensor_name)
                value = type(value)(value.data.to("cpu"), **value.__dict__)
                setattr(module, tensor_name, value)
            # TODO: consider removing used param_parts from state_dict before return

    return error_msgs, offload_index, state_dict_index

# （21）将一个可选的变体名称 (variant) 插入到字符串权重文件名 (weights_name) 中，并返回修改后的文件名
def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:                             # 判断 variant 是否为 None。如果不是 None，则继续执行添加变体的操作
        splits = weights_name.split(".")                # 将 weights_name 按照 . 分隔成一个列表 splits，例如 "model_weights.bin" 会被分隔成 ["model_weights", "bin"]。
        splits = splits[:-1] + [variant] + splits[-1:]  # 通过 splits[:-1] + [variant] + splits[-1:]，在列表的倒数第二个位置插入 variant。
        weights_name = ".".join(splits)                 # 使用 ".".join(splits) 将修改后的列表重新连接成一个字符串，并赋值给 weights_name。

    return weights_name


# 提供了一些针对`torch.nn.Modules`的实用程序, 用作mixin. 被类混合继承
class ModuleUtilsMixin:
    """提供了一些针对`torch.nn.Modules`的实用程序, 用作mixin. 被类混合继承
    定义了14个静态方法和实例方法，用于增强模型的功能，包括内存使用监控、设备和数据类型获取、注意力掩码的创建和操作、参数数量统计等。
    """
    # （1）-（2）静态方法，在模型的每个子模块的前向传播前后（pre_/post_forward）记录内存使用情况。
    @staticmethod
    def _hook_rss_memory_pre_forward(module, *args, **kwargs):
        try:
            import psutil
        except ImportError:
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_pre_forward = mem.rss
        return None

    @staticmethod
    def _hook_rss_memory_post_forward(module, *args, **kwargs):
        try:
            import psutil
        except ImportError:
            raise ImportError("You need to install psutil (pip install psutil) to use memory tracing.")

        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        module.mem_rss_post_forward = mem.rss
        mem_rss_diff = module.mem_rss_post_forward - module.mem_rss_pre_forward
        module.mem_rss_diff = mem_rss_diff + (module.mem_rss_diff if hasattr(module, "mem_rss_diff") else 0)
        return None
    # （3）为模型的所有子模块添加内存使用监控的钩子函数（1）-（2）
    def add_memory_hooks(self):
        """
        在每个子模块forward前后添加一个内存挂钩，以记录内存消耗的增加。
        内存消耗的增加存储在每个模块的`mem_rss_diff`属性中，可以使用`model.reset_memory_hocks_state（）`将其重置为零。
        """
        for module in self.modules():
            module.register_forward_pre_hook(self._hook_rss_memory_pre_forward)
            module.register_forward_hook(self._hook_rss_memory_post_forward)
        self.reset_memory_hooks_state()
    # （4）重置内存钩子的状态，将内存使用差异归零。
    def reset_memory_hooks_state(self):
        """
        Reset the `mem_rss_diff` attribute of each module (see [`~modeling_utils.ModuleUtilsMixin.add_memory_hooks`]).
        """
        for module in self.modules():
            module.mem_rss_diff = 0
            module.mem_rss_post_forward = 0
            module.mem_rss_pre_forward = 0
    # （5）返回模型所在设备的 torch.device。
    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)
    # （6）返回模型参数的数据类型 torch.dtype
    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)
    # （7）反转注意力掩码，将0和1的位置互换。
    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min

        return encoder_extended_attention_mask
    # （8）-（9）创建解码器的扩展注意力掩码，考虑到因果关系（防止解码器看到未来的信息）
    @staticmethod
    def create_extended_attention_mask_for_decoder(input_shape, attention_mask, device=None):
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
        else:
            device = attention_mask.device
        batch_size, seq_length = input_shape
        seq_ids = torch.arange(seq_length, device=device)
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        # in case past_key_values are used we need to add a prefix ones mask to the causal mask
        # causal and attention masks must have same type with pytorch version < 1.3
        causal_mask = causal_mask.to(attention_mask.dtype)

        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = torch.cat(
                [
                    torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
                    causal_mask,
                ],
                axis=-1,
            )

        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        return extended_attention_mask

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = None
    ) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        if not (attention_mask.dim() == 2 and self.config.is_decoder):
            # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
            if device is not None:
                warnings.warn(
                    "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
                )
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask
    # （10）准备头掩码，如果提供了头掩码的话。
    def get_head_mask(
        self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask
    # （11）将头掩码转换为5维张量
    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask
    # （12）统计模型的参数数量，可以选择只计算可训练的参数或排除嵌入层的参数
    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]
            total_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
        else:
            total_parameters = list(self.parameters())

        total_numel = []
        is_loaded_in_4bit = getattr(self, "is_loaded_in_4bit", False)

        if is_loaded_in_4bit:
            if is_bitsandbytes_available():
                import bitsandbytes as bnb
            else:
                raise ValueError(
                    "bitsandbytes is not installed but it seems that the model has been loaded in 4bit precision, something went wrong"
                    " make sure to install bitsandbytes with `pip install bitsandbytes`. You also need a GPU. "
                )

        for param in total_parameters:
            if param.requires_grad or not only_trainable:
                # For 4bit models, we need to multiply the number of parameters by 2 as half of the parameters are
                # used for the 4bit quantization (uint8 tensors are stored)
                if is_loaded_in_4bit and isinstance(param, bnb.nn.Params4bit):
                    if hasattr(param, "element_size"):
                        num_bytes = param.element_size()
                    elif hasattr(param, "quant_storage"):
                        num_bytes = param.quant_storage.itemsize
                    else:
                        num_bytes = 1
                    total_numel.append(param.numel() * 2 * num_bytes)
                else:
                    total_numel.append(param.numel())

        return sum(total_numel)
    # （13）估计模型输入中的总token数量
    def estimate_tokens(self, input_dict: Dict[str, Union[torch.Tensor, Any]]) -> int:
        """
        Helper function to estimate the total number of tokens from the model inputs.

        Args:
            inputs (`dict`): The model inputs.

        Returns:
            `int`: The total number of tokens.
        """
        if not hasattr(self, "warnings_issued"):
            self.warnings_issued = {}
        if self.main_input_name in input_dict:
            return input_dict[self.main_input_name].numel()
        elif "estimate_tokens" not in self.warnings_issued:
            logger.warning(
                "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
            )
            self.warnings_issued["estimate_tokens"] = True
        return 0
    # （5）估算模型在前向和后向传播中使用的浮点运算次数。
    def floating_point_ops(
        self, input_dict: Dict[str, Union[torch.Tensor, Any]], exclude_embeddings: bool = True
    ) -> int:
        """
        Get number of (optionally, non-embeddings) floating-point operations for the forward and backward passes of a
        batch with this transformer model. Default approximation neglects the quadratic dependency on the number of
        tokens (valid if `12 * d_model << sequence_length`) as laid out in [this
        paper](https://arxiv.org/pdf/2001.08361.pdf) section 2.1. Should be overridden for transformers with parameter
        re-use e.g. Albert or Universal Transformers, or if doing long-range modeling with very high sequence lengths.

        Args:
            batch_size (`int`):
                The batch size for the forward pass.

            sequence_length (`int`):
                The number of tokens in each line of the batch.

            exclude_embeddings (`bool`, *optional*, defaults to `True`):
                Whether or not to count embedding and softmax operations.

        Returns:
            `int`: The number of floating-point operations.
        """

        return 6 * self.estimate_tokens(input_dict) * self.num_parameters(exclude_embeddings=exclude_embeddings)

"""所有model的基类，3386行，混合继承自5个父类。含20个类属性，54个类方法
（1）from_pretrained，1134行
（2）_load_pretrained_model，475行
"""
class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PushToHubMixin, PeftAdapterMixin):
    r"""
    所有model的基类。含20个属性，Class attributes可以被派生类derived classes覆盖overridden），54个方法
    [`PreTrainingModel`]负责存储模型的配置，并处理加载、下载和保存模型的方法，以及所有模型共有的一些方法，以：
        -调整输入嵌入的大小，
        -修剪自我注意力的heads。
    """
    config_class = None                         # [`PretrainedConfig`]的一个子类，用作此模型架构的配置类。
    base_model_prefix = ""                      # 在PyTorch模型中加载TensorFlow检查点的python*方法*，参数为：model（PreTrainingModel，加载TensorFlow检查点的模型实例）、config（PreTrainingConfig，与模型关联的配置实例。）、path（str，TensorFlow检查点的路径）
    main_input_name = "input_ids"               # 模型的主要输入的名称（NLP模型通常为“input_ids”，视觉模型为“pixel_values”，语音模型为“input_values”）。
    model_tags = None                           # 存储与模型相关的标签或元数据，例如模型的版本、来源或特定的使用场景。在实例中初始化为空列表后逐个append

    _auto_class = None                          # 用来加载预训练模型，通过from_pretrained()方法可以加载任意Hugging Face中的预训练模型和本地模型
    _no_split_modules = None                    # 模块列表，这些模块在分布式训练或模型并行化时不应该被分割
    _skip_keys_device_placement = None          # 键的列表，这些键对应的模型部分在设备放置时会被忽略。这通常用于那些不需要在特定设备上执行或那些已经通过其他方式管理其设备放置的模型部分。
    _keep_in_fp32_modules = None                # 模块列表，这些模块在模型中应该保持在32位浮点数（fp32）精度。用于混合精度训练

    _keys_to_ignore_on_load_missing = None       # 从缺失键列表（模型中存在的键，但不在检查点中的键）中删除“state_dict”键的“re”模式列表，并避免不必要的警告。
    _keys_to_ignore_on_load_unexpected = None    # 应从我们发现的意外键列表中删除的“state_dict”键的“re”模式列表（检查点内的键，而不是模型内的键），并避免不必要的警告
    _keys_to_ignore_on_save = None               # 保存模型时要忽略的“state_dict”键列表（对于未经过训练但为确定性或绑定变量的键很有用）
    _tied_weights_keys = None                    # 可能与state_dict中的另一个键绑定的“state_dict”键列表。
 
    is_parallelizable = False                    # 指示此模型是否支持模型并行化的标志
    supports_gradient_checkpointing = False      # 是否支持梯度检查点（gradient checkpointing）优化技术，用于减少内存使用，通过在反向传播过程中丢弃一些中间梯度而不是一次性保存所有梯度
    _is_stateful = False                         # 表示模型是否有状态依赖性（stateful）。在某些模型中如循环神经网络（RNN）或其变体中，
                                                 # 隐藏状态（hidden states）可能会跨多个批次或序列步骤传递。如果 _is_stateful 设置为 True，则意味着模型在处理输入时会维护和更新内部状态，这些状态依赖于先前处理的输入数据。

    _supports_flash_attn_2 = False               # 是否支持Flash Attention 2
    _supports_sdpa = False                       # 是否支持SDPA缩放点积注意力

    _supports_cache_class = False                # 是否支持将“Cache”实例设置为“past_key_values”
    _supports_static_cache = False               # 支持“静态缓存”吗
    
    _supports_quantized_cache = False            # 已经支持“QuantoQuantizedCache”实例为“past_key_values”`

    @property  
    def dummy_inputs(self) -> Dict[str, torch.Tensor]:  # （1）返回一个虚拟input_ids张量字典用于前向传递测试。
        """`Dict[str, torch.Tensor]`: 返回一个虚拟input_ids张量字典以执行前向传递。
        内置的装饰器（decorator），它用于将类中的方法转换为一个只读属性。这意味着，尽管这个属性背后实际上是一个方法，但你可以像访问普通属性一样去访问它，而不需要在属性名后加括号（即不需要调用它
        """
        return {"input_ids": torch.tensor(DUMMY_INPUTS)}

    @property
    def framework(self) -> str:                  # （2）返回一个字符串表示模型所基于的框架（PyTorch的“pt”）。
        """
        :str: Identifies that this is a PyTorch model.
        """
        return "pt"
    
    # （3）初始化方法，使用配置对象和其他可选参数初始化模型。
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PretrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        # Save config and origin of the pretrained weights if given in model
        config = self._autoset_attn_implementation(
            config, torch_dtype=torch.get_default_dtype(), check_device_map=False
        )
        self.config = config

        self.name_or_path = config.name_or_path
        self.warnings_issued = {}
        self.generation_config = GenerationConfig.from_model_config(config) if self.can_generate() else None
        # Overwrite the class attribute to make it an instance attribute, so models like
        # `InstructBlipForConditionalGeneration` can dynamically update it without modifying the class attribute
        # when a different component (e.g. language_model) is used.
        self._keep_in_fp32_modules = copy.copy(self.__class__._keep_in_fp32_modules)
    # （4）在每个Transformer模型初始化结束时执行的一种方法，用于执行需要正确初始化模型模块的代码（如权重初始化）。
    def post_init(self):
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()
    # （5）如果模型之前已经量化，则对其进行去量化
    def dequantize(self):
        """
        .
        """
        hf_quantizer = getattr(self, "hf_quantizer", None)

        if hf_quantizer is None:
            raise ValueError("You need to first quantize your model in order to dequantize it")

        return hf_quantizer.dequantize(self)
    # （6）启用渐变检查点以实现向后兼容性。
    def _backward_compatibility_gradient_checkpointing(self):
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()
            # Remove the attribute now that is has been consumed, so it's no saved in the config.
            delattr(self.config, "gradient_checkpointing")
    # （7）为hf hub的模型添加自定义标记，不会重写已有标签
    def add_model_tags(self, tags: Union[List[str], str]) -> None:
        r"""
        Args:
            tags (`Union[List[str], str]`):
                The desired tags to inject in the model

        Examples:

        ```python
        from transformers import AutoModel

        model = AutoModel.from_pretrained("google-bert/bert-base-cased")

        model.add_model_tags(["custom", "custom-bert"])

        # Push the model to your namespace with the name "my-custom-bert".
        model.push_to_hub("my-custom-bert")
        ```
        """
        if isinstance(tags, str):
            tags = [tags]

        if self.model_tags is None:
            self.model_tags = []

        for tag in tags:
            if tag not in self.model_tags:
                self.model_tags.append(tag)
    # （8）从config对象实例化模型，并处理一些参数
    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        工厂方法，用于从配置文件创建模型实例。它主要处理了不同配置参数的设置，并且考虑了使用DeepSpeed ZeRO-3优化技术的情况。所有应该初始化模型的上下文管理器都在这里。
        Args:
            torch_dtype (`torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype.
        """
        torch_dtype = kwargs.pop("torch_dtype", None)
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)

        # 如果需要，覆盖默认dtype
        dtype_orig = None
        if torch_dtype is not None:
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)
        # 不修改_from_config中的配置.
        config = copy.deepcopy(config)

        if config._attn_implementation_internal is not None:
            # 在这种情况下，配置是根据用户设置的attn_implementation参数创建的
            attn_implementation = config._attn_implementation_internal
        else:
            attn_implementation = None

        config._attn_implementation = kwargs.pop("attn_implementation", attn_implementation)
        config = cls._autoset_attn_implementation(
            config,
            use_flash_attention_2=use_flash_attention_2,
            check_device_map=False,
            torch_dtype=torch_dtype,
        )

        if is_deepspeed_zero3_enabled():
            import deepspeed

            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            # 这会立即在所有GPU上对模型进行分区，以避免首先在CPU或每个GPU上复制模型的时间和内存开销
            with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
                model = cls(config, **kwargs)

        else:
            model = cls(config, **kwargs)

        # 如果我们使用`zero3`初始化，则标记为，在模型中添加一个attr，以便我们可以在下游检查问题
        model._transformers_zero3_init_used = is_deepspeed_zero3_enabled()

        # 如果修改了默认dtype，则还原它
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        return model
    # （9）自动设置和分派注意力机制的实现方式。
    @classmethod
    def _autoset_attn_implementation(
        cls,
        config,
        use_flash_attention_2: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
    ):
        """
        自动检查并分派到默认的注意力实现。按优先顺序：:
            1. 在`config.中指定的实现_attn_implementation`（例如，由于from_petrained中的参数attn_implement=“sdpa”）。
            2. DEPRECATED：如果use_flash_attention_2设置为“True”并且“flash_attn”可用，则闪烁注意力。（例如“LlamaFlashAttention”）
            3. SDPA实现（如果可用且受模型类型支持）。（例如“LlamaSdpaAttention”）
            4. 否则默认模型的实现（例如“LlamaAttention”）。
        """
        #这里我们使用config_attn_implementation_internal用于检查用户是否明确设置了注意力实现。
        #属性“PretrainedConfig_attn_implementation永远不会是None，这是为了向后兼容性（总是依赖于“渴望”）。
        #此处的“hasattr”用作某些变压器测试，因为某些原因不调用PretrainedConfig __init__（例如test_no_super_init_config_and_model）

        requested_attn_implementation = None
        if hasattr(config, "_attn_implementation_internal") and config._attn_implementation_internal is not None:
            if config._attn_implementation != "flash_attention_2" and use_flash_attention_2:
                raise ValueError(
                    f'Both attn_implementation="{config._attn_implementation}" and `use_flash_attention_2=True` were used when loading the model, which are not compatible.'
                    ' We recommend to just use `attn_implementation="flash_attention_2"` when loading the model.'
                )

            if config._attn_implementation not in ["eager", "sdpa", "flash_attention_2"]:
                message = f'Specified `attn_implementation="{config._attn_implementation}"` is not supported. The only possible arguments are `attn_implementation="eager"` (manual attention implementation)'
                if cls._supports_flash_attn_2:
                    message += ', `"attn_implementation=flash_attention_2"` (implementation using flash attention 2)'
                if cls._supports_sdpa:
                    message += ', `"attn_implementation=sdpa"` (implementation using torch.nn.functional.scaled_dot_product_attention)'
                raise ValueError(message + ".")

            # If a config is passed with a preset attn_implementation, we skip the automatic dispatch and use the user-provided config, with hard checks that the requested attention implementation is available.
            requested_attn_implementation = config._attn_implementation_internal

        if use_flash_attention_2:
            logger.warning_once(
                'The model was loaded with use_flash_attention_2=True, which is deprecated and may be removed in a future release. Please use `attn_implementation="flash_attention_2"` instead.'
            )
            config._attn_implementation = "flash_attention_2"

        if config._attn_implementation == "flash_attention_2":
            cls._check_and_enable_flash_attn_2(
                config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                hard_check_only=False,
                check_device_map=check_device_map,
            )
        elif requested_attn_implementation in [None, "sdpa"] and not is_torch_xla_available():
            # use_flash_attention_2 takes priority over SDPA, hence SDPA treated in this elif.
            config = cls._check_and_enable_sdpa(
                config,
                hard_check_only=False if requested_attn_implementation is None else True,
            )

            if (
                torch.version.hip is not None
                and config._attn_implementation == "sdpa"
                and torch.cuda.device_count() > 1
            ):
                logger.warning_once(
                    "Using the `SDPA` attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends."
                )
                torch.backends.cuda.enable_flash_sdp(False)
        else:
            config._attn_implementation = "eager"

        return config
    # （10）设置PyTorch的默认数据类型（dtype），并在设置后返回原始的数据类型。这个方法通常在需要以特定的数据类型实例化模型时使用
    @classmethod
    def _set_default_torch_dtype(cls, dtype: torch.dtype) -> torch.dtype:
        """
        更改默认dtype并返回上一个dtype。当想要在特定的dtype下实例化模型时，这是必要的。
        Args:
            dtype (`torch.dtype`):
                a floating dtype to set to.

        Returns:
            `torch.dtype`: the original `dtype` that can be used to restore `torch.set_default_dtype(dtype)` if it was
            modified. If it wasn't, returns `None`.

        Note `set_default_dtype` currently only works with floating-point types and asserts if for example,
        `torch.int64` is passed. So if a non-float `dtype` is passed this functions will throw an exception.
        """
        if not dtype.is_floating_point:
            raise ValueError(
                f"Can't instantiate {cls.__name__} model under dtype={dtype} since it is not a floating point dtype"
            )

        logger.info(f"Instantiating {cls.__name__} model under default dtype {dtype}.")
        dtype_orig = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        return dtype_orig
    # （11）它返回模型的主要部分，即 torch.nn.Module 的实例。这个方法使用了 Python 的内置函数 getattr 来动态获取对象的属性
    @property
    def base_model(self) -> nn.Module:
        """
        `torch.nn.Module`: The main body of the model.
        """
        return getattr(self, self.base_model_prefix, self)
    # （12）用于检测模型是否能够使用 .generate() 方法生成序列
    @classmethod
    def can_generate(cls) -> bool:
        """
        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation.
        # Alternativelly, the model can also have a custom `generate` function.
        if "GenerationMixin" in str(cls.prepare_inputs_for_generation) and "GenerationMixin" in str(cls.generate):
            return False
        return True
    # （13）检查 Flash Attention 2.0 的可用性并确保它与当前模型兼容。如果所有检查通过，并且 hard_check_only 参数为 False，
    # 则此方法将设置配置对象 config 的属性 attn_implementation 为 "flash_attention_2"，以便模型可以初始化正确的注意力模块。
    @classmethod
    def _check_and_enable_flash_attn_2(
        cls,
        config,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
        hard_check_only: bool = False,
    ) -> PretrainedConfig:
        
        if not cls._supports_flash_attn_2:
            raise ValueError(
                f"{cls.__name__} does not support Flash Attention 2.0 yet. Please request to add support where"
                f" the model is hosted, on its model hub page: https://huggingface.co/{config._name_or_path}/discussions/new"
                " or in the Transformers GitHub repo: https://github.com/huggingface/transformers/issues/new"
            )

        if not is_flash_attn_2_available():
            preface = "FlashAttention2 has been toggled on, but it cannot be used due to the following error:"
            install_message = "Please refer to the documentation of https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2 to install Flash Attention 2."

            if importlib.util.find_spec("flash_attn") is None:
                raise ImportError(f"{preface} the package flash_attn seems to be not installed. {install_message}")

            flash_attention_version = version.parse(importlib.metadata.version("flash_attn"))
            if torch.version.cuda:
                if flash_attention_version < version.parse("2.1.0"):
                    raise ImportError(
                        f"{preface} you need flash_attn package version to be greater or equal than 2.1.0. Detected version {flash_attention_version}. {install_message}"
                    )
                else:
                    raise ImportError(f"{preface} Flash Attention 2 is not available. {install_message}")
            elif torch.version.hip:
                if flash_attention_version < version.parse("2.0.4"):
                    raise ImportError(
                        f"{preface} you need flash_attn package version to be greater or equal than 2.0.4. Make sure to have that version installed - detected version {flash_attention_version}. {install_message}"
                    )
                else:
                    raise ImportError(f"{preface} Flash Attention 2 is not available. {install_message}")

        _is_bettertransformer = getattr(cls, "use_bettertransformer", False)

        if _is_bettertransformer:
            raise ValueError(
                "Flash Attention 2 and BetterTransformer API are not compatible. Please make sure to disable BetterTransformers by doing model.reverse_bettertransformer()"
            )

        if torch_dtype is None:
            logger.warning_once(
                "You are attempting to use Flash Attention 2.0 without specifying a torch dtype. This might lead to unexpected behaviour"
            )
        elif torch_dtype is not None and torch_dtype not in [torch.float16, torch.bfloat16]:
            logger.warning_once(
                "Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but"
                f" the current dype in {cls.__name__} is {torch_dtype}. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator,"
                ' or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`'
            )

        # The check `torch.empty(0).device.type != "cuda"` is needed as the model may be initialized after `torch.set_default_device` has been called,
        # or the model may be initialized under the context manager `with torch.device("cuda"):`.
        if check_device_map and device_map is None and torch.empty(0).device.type != "cuda":
            if torch.cuda.is_available():
                logger.warning_once(
                    "You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU"
                    " after initializing it on CPU with `model.to('cuda')`."
                )
            else:
                raise ValueError(
                    "You are attempting to use Flash Attention 2.0 with a model not initialized on GPU and with no GPU available. "
                    "This is not supported yet. Please make sure to have access to a GPU and either initialise the model on a GPU by passing a device_map "
                    "or initialising the model on CPU and then moving it to GPU."
                )
        elif (
            check_device_map
            and device_map is not None
            and isinstance(device_map, dict)
            and ("cpu" in device_map.values() or "disk" in device_map.values())
        ):
            raise ValueError(
                "You are attempting to use Flash Attention 2.0 with a model dispatched on CPU or disk. This is not supported. Please make sure to "
                "initialise the model on a GPU by passing a device_map that contains only GPU devices as keys."
            )
        if not hard_check_only:
            config._attn_implementation = "flash_attention_2"
        return config
    # （14）检查给定模型是否支持通过 torch.nn.functional.scaled_dot_product_attention 实现的注意力机制（SDPA）。
    # 如果所有检查通过，并且 hard_check_only 参数为 False，则此方法将设置配置对象 config 的属性 
    @classmethod
    def _check_and_enable_sdpa(cls, config, hard_check_only: bool = False) -> PretrainedConfig:
        """
        检查给定型号的SDPA的可用性。如果所有检查都通过，并且“hard_check_only”为False，则该方法将配置属性“_attn_implementation”设置为“flash_attention_2”，以便模型能够初始化正确的注意力模块。
        """
        if hard_check_only:
            if not cls._supports_sdpa:
                raise ValueError(
                    f"{cls.__name__} does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention yet."
                    " Please request the support for this architecture: https://github.com/huggingface/transformers/issues/28005. If you believe"
                    ' this error is a bug, please open an issue in Transformers GitHub repository and load your model with the argument `attn_implementation="eager"` meanwhile. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="eager")`'
                )
            if not is_torch_sdpa_available():
                raise ImportError(
                    "PyTorch SDPA requirements in Transformers are not met. Please install torch>=2.1.1."
                )

        if not is_torch_sdpa_available() or not cls._supports_sdpa:
            return config

        _is_bettertransformer = getattr(cls, "use_bettertransformer", False)
        if _is_bettertransformer:
            return config

        if not hard_check_only:
            config._attn_implementation = "sdpa"
        return config
    # （15）-（16）启用、取消模型输入嵌入（embeddings）的梯度计算。
    def enable_input_require_grads(self):
        """
        这在某些特定的训练场景中非常有用，比如在微调（fine-tuning）适配器（adapter）权重时，你可能希望保持模型的其他权重不变。
        """

        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
    
    def disable_input_require_grads(self):
        """
        Removes the `_require_grads_hook`.
        """
        self._require_grads_hook.remove()
    # （17）获取模型输入的embedding
    def get_input_embeddings(self) -> nn.Module:
        """
        Returns:
            `nn.Module`: A torch module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError
    # （18）设置模型的输入嵌入层参数
    def set_input_embeddings(self, value: nn.Module):
        """
        Args:
            value (`nn.Module`): A module mapping vocabulary to hidden states.
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            base_model.set_input_embeddings(value)
        else:
            raise NotImplementedError
    # （19）返回模型的输出嵌入
    def get_output_embeddings(self) -> nn.Module:
        """
        Returns:
            `nn.Module`: A torch module mapping hidden states to vocabulary.
        """
        return None  # Overwrite for models with output embeddings
    # （20）初始化权重：空，必须被派生类覆盖
    def _init_weights(self, module):
        """
        此方法应被派生类覆盖，并且是使用“from_petrained”加载检查点时将调用的唯一初始化方法。
        任何在此函数之外进行初始化的尝试都是无用的，因为torch.nn.init函数都被skip替换了。
        """
        pass
    # （21）如果权重尚未初始化，对其进行初始化
    def _initialize_weights(self, module):
        """
        如果权重尚未初始化，请对其进行初始化。
        """
        if getattr(module, "_is_hf_initialized", False):
            return
        self._init_weights(module)
        module._is_hf_initialized = True
    # （22）在神经网络模型中绑定（共享）权重。这通常用于减少模型的参数数量，提高训练效率
    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.将权重绑定在输入嵌入和输出嵌入之间。
        如果在配置中设置了“torchscript”标志，则无法处理参数共享，因此我们正在克隆权重。
        """
        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_embeddings())

        if getattr(self.config, "is_encoder_decoder", False) and getattr(self.config, "tie_encoder_decoder", False):
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            tied_weights = self._tie_encoder_decoder_weights(
                self.encoder, self.decoder, self.base_model_prefix, "encoder"
            )
            # Setting a dynamic variable instead of `_tied_weights_keys` because it's a class
            # attributed not an instance member, therefore modifying it will modify the entire class
            # Leading to issues on subsequent calls by different tests or subsequent calls.
            self._dynamic_tied_weights_keys = tied_weights

        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()
    # （23）在编码器-解码器模型架构中共享编码器和解码器的权重
    @staticmethod
    def _tie_encoder_decoder_weights(
        encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, base_encoder_name: str
    ):
        uninitialized_encoder_weights: List[str] = []
        tied_weights: List[str] = []
        if decoder.__class__ != encoder.__class__:
            logger.info(
                f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder"
                " weights are correctly initialized."
            )

        def tie_encoder_to_decoder_recursively(
            decoder_pointer: nn.Module,
            encoder_pointer: nn.Module,
            module_name: str,
            base_encoder_name: str,
            uninitialized_encoder_weights: List[str],
            depth=0,
            total_decoder_name="",
            total_encoder_name="",
        ):
            assert isinstance(decoder_pointer, nn.Module) and isinstance(
                encoder_pointer, nn.Module
            ), f"{decoder_pointer} and {encoder_pointer} have to be of type nn.Module"
            if hasattr(decoder_pointer, "weight"):
                assert hasattr(encoder_pointer, "weight")
                encoder_pointer.weight = decoder_pointer.weight
                tied_weights.append(f"{base_encoder_name}{total_encoder_name}.weight")
                if hasattr(decoder_pointer, "bias"):
                    assert hasattr(encoder_pointer, "bias")
                    tied_weights.append(f"{base_encoder_name}{total_encoder_name}.bias")
                    encoder_pointer.bias = decoder_pointer.bias
                return

            encoder_modules = encoder_pointer._modules
            decoder_modules = decoder_pointer._modules
            if len(decoder_modules) > 0:
                assert (
                    len(encoder_modules) > 0
                ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

                all_encoder_weights = {module_name + "/" + sub_name for sub_name in encoder_modules.keys()}
                encoder_layer_pos = 0
                for name, module in decoder_modules.items():
                    if name.isdigit():
                        encoder_name = str(int(name) + encoder_layer_pos)
                        decoder_name = name
                        if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                            encoder_modules
                        ) != len(decoder_modules):
                            # this can happen if the name corresponds to the position in a list module list of layers
                            # in this case the decoder has added a cross-attention that the encoder does not have
                            # thus skip this step and subtract one layer pos from encoder
                            encoder_layer_pos -= 1
                            continue
                    elif name not in encoder_modules:
                        continue
                    elif depth > 500:
                        raise ValueError(
                            "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is"
                            " a circular dependency between two or more `nn.Modules` of your model."
                        )
                    else:
                        decoder_name = encoder_name = name
                    tie_encoder_to_decoder_recursively(
                        decoder_modules[decoder_name],
                        encoder_modules[encoder_name],
                        module_name + "/" + name,
                        base_encoder_name,
                        uninitialized_encoder_weights,
                        depth=depth + 1,
                        total_encoder_name=f"{total_encoder_name}.{encoder_name}",
                        total_decoder_name=f"{total_decoder_name}.{decoder_name}",
                    )
                    all_encoder_weights.remove(module_name + "/" + encoder_name)

                uninitialized_encoder_weights += list(all_encoder_weights)

        # tie weights recursively
        tie_encoder_to_decoder_recursively(
            decoder, encoder, base_model_prefix, base_encoder_name, uninitialized_encoder_weights
        )

        if len(uninitialized_encoder_weights) > 0:
            logger.warning(
                f"The following encoder weights were not tied to the decoder {uninitialized_encoder_weights}"
            )
        return tied_weights
    # （24）根据是否使用TorchScript，绑定或克隆模块权重
    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """根据我们是否使用TorchScript，绑定或克隆模块权重"""
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
            output_embeddings.out_features = input_embeddings.num_embeddings
    # （25）用于确定在使用设备映射（device_map）时模型中不应该被分割的模块
    def _get_no_split_modules(self, device_map: str):
        """
        Get the modules of the model that should not be spit when using device_map. We iterate through the modules to
        get the underlying `_no_split_modules`.

        Args:
            device_map (`str`):
                The device map value. Options are ["auto", "balanced", "balanced_low_0", "sequential"]

        Returns:
            `List[str]`: List of modules that should not be split
        """
        _no_split_modules = set()
        modules_to_check = [self]
        while len(modules_to_check) > 0:
            module = modules_to_check.pop(-1)
            # if the module does not appear in _no_split_modules, we also check the children
            if module.__class__.__name__ not in _no_split_modules:
                if isinstance(module, PreTrainedModel):
                    if module._no_split_modules is None:
                        raise ValueError(
                            f"{module.__class__.__name__} does not support `device_map='{device_map}'`. To implement support, the model "
                            "class needs to implement the `_no_split_modules` attribute."
                        )
                    else:
                        _no_split_modules = _no_split_modules | set(module._no_split_modules)
                modules_to_check += list(module.children())
        return list(_no_split_modules)
    # （26）—（28）调整模型中输入令牌（token）嵌入矩阵的大小。这个方法通常在深度学习模型中使用，特别是在需要改变词汇表大小或者优化硬件性能时
    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        """
        如果`new_num_tokens！=config.vocab_size，则调整模型的输入令牌嵌入矩阵的大小`。
        如果模型类有一个`tie_weights（）`方法，则负责在之后绑定权重嵌入。
        Arguments:
            new_num_tokens (`int`, *optional*):
                The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value.If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Since we are basically resuing the same old embeddings with new weight values, gathering is required
        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            with deepspeed.zero.GatheredParameters(model_embeds.weight, modifier_rank=None):
                vocab_size = model_embeds.weight.shape[0]
        else:
            vocab_size = model_embeds.weight.shape[0]

        # Update base model and current model config
        if hasattr(self.config, "text_config"):
            self.config.text_config.vocab_size = vocab_size
        else:
            self.config.vocab_size = vocab_size
        self.vocab_size = vocab_size

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of)
        if hasattr(old_embeddings, "_hf_hook"):
            hook = old_embeddings._hf_hook
            add_hook_to_module(new_embeddings, hook)
        old_embeddings_requires_grad = old_embeddings.weight.requires_grad
        new_embeddings.requires_grad_(old_embeddings_requires_grad)
        self.set_input_embeddings(new_embeddings)
        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None

        # Update new_num_tokens with the actual size of new_embeddings
        if pad_to_multiple_of is not None:
            if is_deepspeed_zero3_enabled() and not is_quantized:
                import deepspeed

                with deepspeed.zero.GatheredParameters(new_embeddings.weight, modifier_rank=None):
                    new_num_tokens = new_embeddings.weight.shape[0]
            else:
                new_num_tokens = new_embeddings.weight.shape[0]

        # if word embeddings are not tied, make sure that lm head is resized as well
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            old_lm_head = self.get_output_embeddings()
            if isinstance(old_lm_head, torch.nn.Embedding):
                new_lm_head = self._get_resized_embeddings(old_lm_head, new_num_tokens)
            else:
                new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
            if hasattr(old_lm_head, "_hf_hook"):
                hook = old_lm_head._hf_hook
                add_hook_to_module(new_lm_head, hook)
            old_lm_head_requires_grad = old_lm_head.weight.requires_grad
            new_lm_head.requires_grad_(old_lm_head_requires_grad)
            self.set_output_embeddings(new_lm_head)

        return self.get_input_embeddings()

    def _get_resized_embeddings(
        self,
        old_embeddings: nn.Embedding,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> nn.Embedding:
        """
        从提供的令牌嵌入模块构建调整大小的嵌入模块。增加大小将在末尾添加新初始化的向量。减小大小将从末尾删除向量
        Args:
            old_embeddings (`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc


        Return:
            `torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            `new_num_tokens` is `None`
        """

        if pad_to_multiple_of is not None:
            if not isinstance(pad_to_multiple_of, int):
                raise ValueError(
                    f"Asking to pad the embedding matrix to a multiple of `{pad_to_multiple_of}`, which is not and integer. Please make sure to pass an integer"
                )
            if new_num_tokens is None:
                new_num_tokens = old_embeddings.weight.shape[0]
            new_num_tokens = ((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        else:
            logger.info(
                "You are resizing the embedding layer without providing a `pad_to_multiple_of` parameter. This means that the new embedding"
                f" dimension will be {new_num_tokens}. This might induce some performance reduction as *Tensor Cores* will not be available."
                " For more details about this, or help on choosing the correct value for resizing, refer to this guide:"
                " https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc"
            )

        if new_num_tokens is None:
            return old_embeddings

        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
                old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens and not is_deepspeed_zero3_enabled():
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that `old_embeddings` are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
        # because the shape of the new embedding layer is used across various modeling files
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
        # to errors when training.
        new_embeddings = nn.Embedding(
            new_num_tokens,
            old_embedding_dim,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype,
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)

        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            params = [old_embeddings.weight, new_embeddings.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        else:
            new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        # Replace weights in old_embeddings and return to maintain the same embedding type.
        # This ensures correct functionality when a Custom Embedding class is passed as input.
        # The input and output embedding types remain consistent. (c.f. https://github.com/huggingface/transformers/pull/31979)
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            params = [old_embeddings.weight, new_embeddings.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                old_embeddings.weight = new_embeddings.weight
                old_embeddings.num_embeddings = new_embeddings.weight.data.shape[0]

                # If the new number of tokens is smaller than the original `padding_idx`, the `padding_idx`
                # will be set to `None` in the resized embeddings.
                if old_embeddings.padding_idx is not None and (new_num_tokens - 1) < old_embeddings.padding_idx:
                    old_embeddings.padding_idx = None
        else:
            old_embeddings.weight.data = new_embeddings.weight.data
            old_embeddings.num_embeddings = new_embeddings.weight.data.shape[0]
            if old_embeddings.padding_idx is not None and (new_num_tokens - 1) < old_embeddings.padding_idx:
                old_embeddings.padding_idx = None

        return old_embeddings
    # （29）-（30）调整模型中输出层lm-head的大小。
    def _get_resized_lm_head(
        self, old_lm_head: nn.Linear, new_num_tokens: Optional[int] = None, transposed: Optional[bool] = False
    ) -> nn.Linear:
        """
        Build a resized Linear Module from a provided old Linear Module. Increasing the size will add newly initialized
        vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_lm_head (`torch.nn.Linear`):
                Old lm head liner layer to be resized.
            new_num_tokens (`int`, *optional*):
                New number of tokens in the linear matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or `None`, just returns a pointer to the input tokens
                `torch.nn.Linear` module of the model without doing anything. transposed (`bool`, *optional*, defaults
                to `False`): Whether `old_lm_head` is transposed or not. If True `old_lm_head.size()` is `lm_head_dim,
                vocab_size` else `vocab_size, lm_head_dim`.

        Return:
            `torch.nn.Linear`: Pointer to the resized Linear Module or the old Linear Module if `new_num_tokens` is
            `None`
        """
        if new_num_tokens is None:
            return old_lm_head

        is_quantized = hasattr(self, "hf_quantizer") and self.hf_quantizer is not None
        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_lm_head.weight, modifier_rank=None):
                old_num_tokens, old_lm_head_dim = (
                    old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
                )
        else:
            old_num_tokens, old_lm_head_dim = (
                old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
            )

        if old_num_tokens == new_num_tokens and not is_deepspeed_zero3_enabled():
            return old_lm_head

        if not isinstance(old_lm_head, nn.Linear):
            raise TypeError(
                f"Old language model head is of type {type(old_lm_head)}, which is not an instance of {nn.Linear}. You"
                " should either use a different resize function or make sure that `old_lm_head` are an instance of"
                f" {nn.Linear}."
            )

        # Build new lm head
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = old_lm_head.bias is not None

        # When using DeepSpeed ZeRO-3, we shouldn't create new embeddings with DeepSpeed init
        # because the shape of the new embedding layer is used across various modeling files
        # as well as to update config vocab size. Shape will be 0 when using DeepSpeed init leading
        # to errors when training.
        new_lm_head = nn.Linear(
            *new_lm_head_shape,
            bias=has_new_lm_head_bias,
            device=old_lm_head.weight.device,
            dtype=old_lm_head.weight.dtype,
        )

        # initialize new lm head (in particular added tokens)
        self._init_weights(new_lm_head)

        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)

        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            params = [old_lm_head.weight, old_lm_head.bias, new_lm_head.weight, new_lm_head.bias]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                self._copy_lm_head_original_to_resized(
                    new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
                )
        else:
            self._copy_lm_head_original_to_resized(
                new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
            )

        return new_lm_head

    def _copy_lm_head_original_to_resized(
        self, new_lm_head, old_lm_head, num_tokens_to_copy, transposed, has_new_lm_head_bias
    ):
        # Copy old lm head weights to new lm head
        if not transposed:
            new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        else:
            new_lm_head.weight.data[:, :num_tokens_to_copy] = old_lm_head.weight.data[:, :num_tokens_to_copy]

        # Copy bias weights to new lm head
        if has_new_lm_head_bias:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]
    # （31）-（32）调整模型中位置编码层的大小。
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        raise NotImplementedError(
            f"`resize_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )

    def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
        raise NotImplementedError(
            f"`get_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `modeling_{self.__class__.__module__}.py`"
        )
    # （33）有效的初始化权重
    def init_weights(self):
        """
        如果需要，修剪并初始化权重。如果使用自定义的“PreTrainingModel”，则需要在“_init_weights”中实现任何初始化逻辑`
        """
        # Prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)

        if _init_weights:
            # Initialize weights
            self.apply(self._initialize_weights)

            # 当不初始化所有权重时，应跳过绑定权重，因为from_petrained（…）无论如何都会调用绑定权重
            self.tie_weights()
    # （34）剪枝：基础模型的注意力头。剪枝是一种压缩模型的技术，通过移除一些注意力头来减少模型的复杂性和大小，同时可能还能提高效率。
    def prune_heads(self, heads_to_prune: Dict[int, List[int]]):
        """
        Arguments:
            heads_to_prune (`Dict[int, List[int]]`):
                Dictionary with keys being selected layer indices (`int`) and associated values being the list of heads
                to prune in said layer (list of `int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads 0 and 2 on
                layer 1 and heads 2 and 3 on layer 2.
        """
        # save new sets of pruned heads as union of previously stored pruned heads and newly pruned heads
        for layer, heads in heads_to_prune.items():
            union_heads = set(self.config.pruned_heads.get(layer, [])) | set(heads)
            self.config.pruned_heads[layer] = list(union_heads)  # Unfortunately we have to store it as list for JSON

        self.base_model._prune_heads(heads_to_prune)
    # （35）-（38）激活、禁止、设置模型的梯度检查点（gradient checkpointing）功能，判断是否开启这一功能。
    #  梯度检查点是一种节省内存的技术，可以在深度学习训练过程中减少内存消耗，通过在正向传播中丢弃一些中间梯度，然后在反向传播时重新计算它们来实现。
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".

        We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
        the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

        Args:
            gradient_checkpointing_kwargs (dict, *optional*):
                Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": True}

        gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)

        # For old GC format (transformers < 4.35.0) for models that live on the Hub
        # we will fall back to the overwritten `_set_gradient_checkpointing` method
        _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters

        if not _is_using_old_format:
            self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
        else:
            self.apply(partial(self._set_gradient_checkpointing, value=True))
            logger.warning(
                "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
            )

        if getattr(self, "_hf_peft_config_loaded", False):
            # When using PEFT + gradient checkpointing + Trainer we need to make sure the input has requires_grad=True
            # we do it also on PEFT: https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L334
            # When training with PEFT, only LoRA layers will have requires grad set to True, but the output of frozen layers need to propagate
            # the gradients to make sure the gradient flows.
            self.enable_input_require_grads()
    
    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint):
        is_gradient_checkpointing_set = False

        # Apply it on the top-level module in case the top-level modules supports it
        # for example, LongT5Stack inherits from `PreTrainedModel`.
        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self.supports_gradient_checkpointing:
            # For old GC format (transformers < 4.35.0) for models that live on the Hub
            # we will fall back to the overwritten `_set_gradient_checkpointing` methid
            _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters
            if not _is_using_old_format:
                self._set_gradient_checkpointing(enable=False)
            else:
                logger.warning(
                    "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                    "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
                )
                self.apply(partial(self._set_gradient_checkpointing, value=False))

        if getattr(self, "_hf_peft_config_loaded", False):
            self.disable_input_require_grads()

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.modules())
    # （39）保存预训练模型及其配置文件到指定目录，以便之后可以使用类方法from_pretrained重新加载模型
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~PreTrainedModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            state_dict (nested dictionary of `torch.Tensor`):
                The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only
                save parts of the model or if special precautions need to be taken when recovering the state dictionary
                of a model (like when using model parallelism).
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
                We default it to 5GB in order for models to be able to run easily on free-tier google colab instances
                without CPU OOM issues.

                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>

            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            save_peft_format (`bool`, *optional*, defaults to `True`):
                For backward compatibility with PEFT library, in case adapter weights are attached to the model, all
                keys of the state dict of adapters needs to be pre-pended with `base_model.model`. Advanced users can
                disable this behaviours by setting `save_peft_format` to `False`.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        use_auth_token = kwargs.pop("use_auth_token", None)
        ignore_metadata_errors = kwargs.pop("ignore_metadata_errors", False)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

        _hf_peft_config_loaded = getattr(self, "_hf_peft_config_loaded", False)

        hf_quantizer = getattr(self, "hf_quantizer", None)
        quantization_serializable = (
            hf_quantizer is not None and isinstance(hf_quantizer, HfQuantizer) and hf_quantizer.is_serializable
        )

        if hf_quantizer is not None and not _hf_peft_config_loaded and not quantization_serializable:
            raise ValueError(
                f"The model is quantized with {hf_quantizer.quantization_config.quant_method} and is not serializable - check out the warnings from"
                " the logger on the traceback to understand the reason why the quantized model is not serializable."
            )

        if "save_config" in kwargs:
            warnings.warn(
                "`save_config` is deprecated and will be removed in v5 of Transformers. Use `is_main_process` instead."
            )
            is_main_process = kwargs.pop("save_config")
        if safe_serialization and not is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)

        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.torch_dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self.config)

        # Save the config
        if is_main_process:
            if not _hf_peft_config_loaded:
                model_to_save.config.save_pretrained(save_directory)
            if self.can_generate():
                # generation config built from the model config + the model config holds generation kwargs -> generate
                # may revert to legacy behavior if the two don't match
                if (
                    model_to_save.generation_config._from_model_config
                    and model_to_save.config._has_non_default_generation_parameters()
                ):
                    new_generation_config = GenerationConfig.from_model_config(model_to_save.config)
                    if new_generation_config != model_to_save.generation_config:
                        logger.warning(
                            "Your generation config was originally created from the model config, but the model "
                            "config has changed since then. Unless you pass the `generation_config` argument to this "
                            "model's `generate` calls, they will revert to the legacy behavior where the base "
                            "`generate` parameterization is loaded from the model config instead. "
                            "To avoid this behavior and this warning, we recommend you to overwrite the generation "
                            "config model attribute before calling the model's `save_pretrained`, preferably also "
                            "removing any generation kwargs from the model config. This warning will be raised to an "
                            "exception in v4.41."
                        )
                model_to_save.generation_config.save_pretrained(save_directory)

            if _hf_peft_config_loaded:
                logger.info(
                    "Detected adapters on the model, saving the model in the PEFT format, only adapter weights will be saved."
                )
                state_dict = model_to_save.get_adapter_state_dict()

                if save_peft_format:
                    logger.info(
                        "To match the expected format of the PEFT library, all keys of the state dict of adapters will be pre-pended with `base_model.model`."
                    )
                    peft_state_dict = {}
                    for key, value in state_dict.items():
                        peft_state_dict[f"base_model.model.{key}"] = value
                    state_dict = peft_state_dict

                active_adapter = self.active_adapters()

                if len(active_adapter) > 1:
                    raise ValueError(
                        "Multiple active adapters detected, saving multiple active adapters is not supported yet. You can save adapters separately one by one "
                        "by iteratively calling `model.set_adapter(adapter_name)` then `model.save_pretrained(...)`"
                    )
                active_adapter = active_adapter[0]

                current_peft_config = self.peft_config[active_adapter]
                current_peft_config.save_pretrained(save_directory)

        # for offloaded modules
        module_map = {}

        # Save the model
        if state_dict is None:
            # if any model parameters are offloaded, make module map
            if (
                hasattr(self, "hf_device_map")
                and len(set(self.hf_device_map.values())) > 1
                and ("cpu" in self.hf_device_map.values() or "disk" in self.hf_device_map.values())
            ):
                warnings.warn(
                    "Attempting to save a model with offloaded modules. Ensure that unallocated cpu memory exceeds the `shard_size` (5GB default)"
                )
                for name, module in model_to_save.named_modules():
                    if name == "":
                        continue
                    module_state_dict = module.state_dict()

                    for key in module_state_dict:
                        module_map[name + f".{key}"] = module
            state_dict = model_to_save.state_dict()

        # Translate state_dict from smp to hf if saving with smp >= 1.10
        if IS_SAGEMAKER_MP_POST_1_10:
            for smp_to_hf, _ in smp.state.module_manager.translate_functions:
                state_dict = smp_to_hf(state_dict)

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            for ignore_key in self._keys_to_ignore_on_save:
                if ignore_key in state_dict.keys():
                    del state_dict[ignore_key]
        if safe_serialization:
            # Safetensors does not allow tensor aliasing.
            # We're going to remove aliases before saving
            ptrs = collections.defaultdict(list)
            for name, tensor in state_dict.items():
                # Sometimes in the state_dict we have non-tensor objects.
                # e.g. in bitsandbytes we have some `str` objects in the state_dict
                if isinstance(tensor, torch.Tensor):
                    ptrs[id_tensor_storage(tensor)].append(name)
                else:
                    # In the non-tensor case, fall back to the pointer of the object itself
                    ptrs[id(tensor)].append(name)

            # These are all the pointers of shared tensors
            if hasattr(self, "hf_device_map"):
                # if the model has offloaded parameters, we must check using find_tied_parameters()
                tied_params = find_tied_parameters(self)
                if tied_params:
                    tied_names = tied_params[0]
                    shared_ptrs = {
                        ptr: names for ptr, names in ptrs.items() if any(name in tied_names for name in names)
                    }
                else:
                    shared_ptrs = {}
            else:
                shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

            # Recursively descend to find tied weight keys
            _tied_weights_keys = _get_tied_weight_keys(self)
            error_names = []
            to_delete_names = set()
            for names in shared_ptrs.values():
                # Removing the keys which are declared as known duplicates on
                # load. This allows to make sure the name which is kept is consistent.
                if _tied_weights_keys is not None:
                    found = 0
                    for name in sorted(names):
                        matches_pattern = any(re.search(pat, name) for pat in _tied_weights_keys)
                        if matches_pattern and name in state_dict:
                            found += 1
                            if found < len(names):
                                to_delete_names.add(name)
            # We are entering a place where the weights and the transformers configuration do NOT match.
            shared_names, disjoint_names = _find_disjoint(shared_ptrs.values(), state_dict)
            # Those are actually tensor sharing but disjoint from each other, we can safely clone them
            # Reloaded won't have the same property, but it shouldn't matter in any meaningful way.
            for name in disjoint_names:
                state_dict[name] = state_dict[name].clone()

            # When not all duplicates have been cleaned, still remove those keys, but put a clear warning.
            # If the link between tensors was done at runtime then `from_pretrained` will not get
            # the key back leading to random tensor. A proper warning will be shown
            # during reload (if applicable), but since the file is not necessarily compatible with
            # the config, better show a proper warning.
            shared_names, identical_names = _find_identical(shared_names, state_dict)
            # delete tensors that have identical storage
            for inames in identical_names:
                known = inames.intersection(to_delete_names)
                for name in known:
                    del state_dict[name]
                unknown = inames.difference(to_delete_names)
                if len(unknown) > 1:
                    error_names.append(unknown)

            if shared_names:
                error_names.append(set(shared_names))

            if len(error_names) > 0:
                raise RuntimeError(
                    f"The weights trying to be saved contained shared tensors {error_names} that are mismatching the transformers base configuration. Try saving using `safe_serialization=False` or remove this tensor sharing.",
                )

        # Shard the model if it is too big.
        if not _hf_peft_config_loaded:
            weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
            weights_name = _add_variant(weights_name, variant)
        else:
            weights_name = ADAPTER_SAFE_WEIGHTS_NAME if safe_serialization else ADAPTER_WEIGHTS_NAME

        filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
        )
        # Save index if sharded
        index = None
        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")

            # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
            filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
            reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in state_dict_split.filename_to_tensors.keys()
                and is_main_process
                and reg.fullmatch(filename_no_suffix) is not None
            ):
                os.remove(full_filename)
        # Save the model
        filename_to_tensors = state_dict_split.filename_to_tensors.items()
        if module_map:
            filename_to_tensors = logging.tqdm(filename_to_tensors, desc="Saving checkpoint shards")
        for shard_file, tensors in filename_to_tensors:
            shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}
            # remake shard with onloaded parameters if necessary
            if module_map:
                if accelerate_version < version.parse("0.31"):
                    raise ImportError(
                        f"You need accelerate version to be greater or equal than 0.31 to save models with offloaded parameters. Detected version {accelerate_version}. "
                        f"Please upgrade accelerate with `pip install -U accelerate`"
                    )
                # init state_dict for this shard
                shard_state_dict = {name: "" for name in shard}
                for module_name in shard:
                    module = module_map[module_name]
                    # update state dict with onloaded parameters
                    shard_state_dict = get_state_dict_from_offload(module, module_name, shard_state_dict)

                # assign shard to be the completed state dict
                shard = shard_state_dict
                del shard_state_dict
                gc.collect()

            if safe_serialization:
                # At some point we will need to deal better with save_function (used for TPU and other distributed
                # joyfulness), but for now this enough.
                safe_save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})
            else:
                save_function(shard, os.path.join(save_directory, shard_file))

        if index is None:
            path_to_weights = os.path.join(save_directory, weights_name)
            logger.info(f"Model weights saved in {path_to_weights}")
        else:
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(state_dict_split.filename_to_tensors)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

        if push_to_hub:
            # Eventually create an empty model card
            model_card = create_and_tag_model_card(
                repo_id, self.model_tags, token=token, ignore_metadata_errors=ignore_metadata_errors
            )

            # Update model card if needed:
            model_card.save(os.path.join(save_directory, "README.md"))

            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
            )
    # （40）将训练好的模型推送到Hub
    @wraps(PushToHubMixin.push_to_hub)
    def push_to_hub(self, *args, **kwargs):
        '''
        在调用PushToHubMixin.push_to_hub方法之前，处理和更新模型的标签（tags）。
        如果提供了新的标签，并且这些标签不在现有的标签列表中，则将它们添加到列表中。
        然后，使用更新后的标签列表调用原始的push_to_hub方法。这种方法在机器学习模型的版本控制和共享中非常有用，尤其是在使用Hugging Face的Transformers库时。
        '''
        tags = self.model_tags if self.model_tags is not None else []

        tags_kwargs = kwargs.get("tags", [])
        if isinstance(tags_kwargs, str):
            tags_kwargs = [tags_kwargs]

        for tag in tags_kwargs:
            if tag not in tags:
                tags.append(tag)

        if tags:
            kwargs["tags"] = tags
        return super().push_to_hub(*args, **kwargs)
    # （41）用于计算模型的内存占用量，以字节为单位。
    def get_memory_footprint(self, return_buffers=True):
        r"""
        Get the memory footprint of a model. This will return the memory footprint of the current model in bytes.
        Useful to benchmark the memory footprint of the current model and design some tests. Solution inspired from the
        PyTorch discussions: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2

        Arguments:
            return_buffers (`bool`, *optional*, defaults to `True`):
                Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers
                are tensors that do not require gradients and not registered as parameters. E.g. mean and std in batch
                norm layers. Please see: https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266/2
        """
        mem = sum([param.nelement() * param.element_size() for param in self.parameters()])
        if return_buffers:
            mem_bufs = sum([buf.nelement() * buf.element_size() for buf in self.buffers()])
            mem = mem + mem_bufs
        return mem
    # （42）-（45）在PyTorch模型中重写基类的cuda、to、half、float方法。在涉及到量化模型的情况下进行限制
    @wraps(torch.nn.Module.cuda)
    def cuda(self, *args, **kwargs):
        if getattr(self, "quantization_method", None) == QuantizationMethod.HQQ:
            raise ValueError("`.cuda` is not supported for HQQ-quantized models.")
        # Checks if the model has been loaded in 8-bit
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            raise ValueError(
                "Calling `cuda()` is not supported for `4-bit` or `8-bit` quantized models. Please use the model as it is, since the"
                " model has already been set to the correct devices and casted to the correct `dtype`."
            )
        else:
            return super().cuda(*args, **kwargs)

    @wraps(torch.nn.Module.to)
    def to(self, *args, **kwargs):
        if getattr(self, "quantization_method", None) == QuantizationMethod.HQQ:
            raise ValueError("`.to` is not supported for HQQ-quantized models.")
        # Checks if the model has been loaded in 8-bit
        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            raise ValueError(
                "`.to` is not supported for `4-bit` or `8-bit` bitsandbytes models. Please use the model as it is, since the"
                " model has already been set to the correct devices and casted to the correct `dtype`."
            )
        elif getattr(self, "quantization_method", None) == QuantizationMethod.GPTQ:
            # For GPTQ models, we prevent users from casting the model to another dytpe to restrict unwanted behaviours.
            # the correct API should be to load the model with the desired dtype directly through `from_pretrained`.
            dtype_present_in_args = False

            if "dtype" not in kwargs:
                for arg in args:
                    if isinstance(arg, torch.dtype):
                        dtype_present_in_args = True
                        break
            else:
                dtype_present_in_args = True

            if dtype_present_in_args:
                raise ValueError(
                    "You cannot cast a GPTQ model in a new `dtype`. Make sure to load the model using `from_pretrained` using the desired"
                    " `dtype` by passing the correct `torch_dtype` argument."
                )
        return super().to(*args, **kwargs)

    def half(self, *args):
        # Checks if the model is quantized
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.half()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been casted to the correct `dtype`."
            )
        else:
            return super().half(*args)

    def float(self, *args):
        # Checks if the model is quantized
        if getattr(self, "is_quantized", False):
            raise ValueError(
                "`.float()` is not supported for quantized model. Please use the model as it is, since the"
                " model has already been casted to the correct `dtype`."
            )
        else:
            return super().float(*args)
    
    # （46）从一个预训练模型配置文件中实例化一个预训练的PyTorch模型
    '''如果提供了配置对象，则使用该配置来初始化模型。根据pretrained_model_name_or_path和config确定模型权重的来源。
    根据文件类型（PyTorch、TensorFlow、Flax或safetensors）加载模型权重。如果设置了low_cpu_mem_usage，则尝试以较低的CPU内存使用量加载模型。
    如果模型是量化的，将应用量化配置并相应地加载权重。如果模型包含适配器，将加载适配器权重。最终，模型被设置为评估模式并返回。'''
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ) -> "PreTrainedModel":
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        If model weights are the same precision as the base model (and is a supported model), weights will be lazily loaded
        in using the `meta` device and brought into memory once an input is passed through that layer regardless of
        `low_cpu_mem_usage`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                    - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                      `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                      `True`.
                    - `None` if you are both providing the configuration and state dictionary (resp. with keyword
                      arguments `config` and `state_dict`).
            model_args (sequence of positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.
            config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
                Can be either:

                    - an instance of a class derived from [`PretrainedConfig`],
                    - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].

                Configuration for the model to use instead of an automatically loaded configuration. Configuration can
                be automatically loaded when:

                    - The model is a model provided by the library (loaded with the *model id* string of a pretrained
                      model).
                    - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
                      save directory.
                    - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
                      configuration JSON file named *config.json* is found in the directory.
            state_dict (`Dict[str, torch.Tensor]`, *optional*):
                A state dictionary to use instead of a state dictionary loaded from saved weights file.

                This option can be used if you want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
                [`~PreTrainedModel.from_pretrained`] is not a simpler option.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            from_tf (`bool`, *optional*, defaults to `False`):
                Load the model weights from a TensorFlow checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file (see docstring of
                `pretrained_model_name_or_path` argument).
            ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
                Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
                as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
                checkpoint with 3 labels).
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

                </Tip>

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            _fast_init(`bool`, *optional*, defaults to `True`):
                Whether or not to disable fast initialization.

                <Tip warning={true}>

                One should only disable *_fast_init* to ensure backwards compatibility with `transformers.__version__ <
                4.6.0` for seeded model initialization. This argument will be removed at the next major version. See
                [pull request 11471](https://github.com/huggingface/transformers/pull/11471) for more information.

                </Tip>
            attn_implementation (`str`, *optional*):
                The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

            > Parameters for big model inference

            low_cpu_mem_usage(`bool`, *optional*):
                Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Generally should be combined with a `device_map` (such as `"auto"`) for best results.
                This is an experimental feature and a subject to change at any moment.
                </Tip>
                    If the model weights are in the same precision as the model loaded in, `low_cpu_mem_usage` (without
                    `device_map`) is redundant and will not provide any benefit in regards to CPU memory usage. However,
                    this should still be enabled if you are passing in a `device_map`.
                </Tip>
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under a specific `dtype`. The different options
                are:

                1. `torch.float16` or `torch.bfloat16` or `torch.float`: load in a specified
                  `dtype`, ignoring the model's `config.torch_dtype` if one exists. If not specified
                  - the model will get loaded in `torch.float` (fp32).

                2. `"auto"` - A `torch_dtype` entry in the `config.json` file of the model will be
                  attempted to be used. If this entry isn't found then next check the `dtype` of the first weight in
                  the checkpoint that's of a floating point type and use that as `dtype`. This will load the model
                  using the `dtype` it was saved in at the end of the training. It can't be used as an indicator of how
                  the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.

                3. A string that is a valid `torch.dtype`. E.g. "float32" loads the model in `torch.float32`, "float16" loads in `torch.float16` etc.

                <Tip>

                For some models the `dtype` they were trained in is unknown - you may try to check the model's paper or
                reach out to the authors and ask them to add this information to the model's card and to insert the
                `torch_dtype` entry in `config.json` on the hub.

                </Tip>

            device_map (`str` or `Dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank
                like `1`) on which the model will be allocated, the device map will map the entire model to this
                device. Passing `device_map = 0` means put the whole model on GPU 0.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier to maximum memory. Will default to the maximum memory available for each
                GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
            offload_state_dict (`bool`, *optional*):
                If `True`, will temporarily offload the CPU state dict to the hard drive to avoid getting out of CPU
                RAM if the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to
                `True` when there is some disk offload.
            offload_buffers (`bool`, *optional*):
                Whether or not to offload the buffers with the model parameters.
            quantization_config (`Union[QuantizationConfigMixin,Dict]`, *optional*):
                A dictionary of configuration parameters or a QuantizationConfigMixin object for quantization (e.g
                bitsandbytes, gptq). There may be other quantization-related kwargs, including `load_in_4bit` and
                `load_in_8bit`, which are parsed by QuantizationConfigParser. Supported only for bitsandbytes
                quantizations and not preferred. consider inserting all such arguments into quantization_config
                instead.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            variant (`str`, *optional*):
                If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. `variant` is
                ignored when using `from_tf` or `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                Whether or not to use `safetensors` checkpoints. Defaults to `None`. If not specified and `safetensors`
                is not installed, it will be set to `False`.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
                automatically loaded:

                    - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
                      underlying model's `__init__` method (we assume all relevant updates to the configuration have
                      already been done)
                    - If a configuration is not provided, `kwargs` will be first passed to the configuration class
                      initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
                      corresponds to a configuration attribute will be used to override said attribute with the
                      supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
                      will be passed to the underlying model's `__init__` function.

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>

        Examples:

        ```python
        >>> from transformers import BertConfig, BertModel

        >>> # Download model and configuration from huggingface.co and cache.
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        >>> model = BertModel.from_pretrained("./test/saved_model/")
        >>> # Update configuration during loading.
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", output_attentions=True)
        >>> assert model.config.output_attentions == True
        >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
        >>> config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
        >>> model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)
        >>> # Loading from a Flax checkpoint file instead of a PyTorch model (slower)
        >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", from_flax=True)
        ```

        * `low_cpu_mem_usage` algorithm:

        This is an experimental function that loads the model using ~1x model size CPU memory

        Here is how it works:

        1. save which state_dict keys we have
        2. drop state_dict before the model is created, since the latter takes 1x model size CPU memory
        3. after the model has been instantiated switch to the meta device all params/buffers that
        are going to be replaced from the loaded state_dict
        4. load state_dict 2nd time
        5. replace the params/buffers from the state_dict

        Currently, it can't handle deepspeed ZeRO stage 3 and ignores loading errors

        """
        state_dict = kwargs.pop("state_dict", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        _ = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        offload_buffers = kwargs.pop("offload_buffers", False)
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        quantization_config = kwargs.pop("quantization_config", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        variant = kwargs.pop("variant", None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", {})
        adapter_name = kwargs.pop("adapter_name", "default")
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)

        gguf_file = kwargs.pop("gguf_file", None)
        # Cache path to the GGUF file
        gguf_path = None

        if is_fsdp_enabled():
            low_cpu_mem_usage = True

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None and adapter_kwargs is not None and "token" not in adapter_kwargs:
            adapter_kwargs["token"] = token

        if use_safetensors is None and not is_safetensors_available():
            use_safetensors = False
        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
                " ignored."
            )

        if gguf_file is not None and not is_accelerate_available():
            raise ValueError("accelerate is required when loading a GGUF file `pip install accelerate`.")

        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):
                # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    CONFIG_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, "_commit_hash", None)

        if is_peft_available():
            _adapter_model_path = adapter_kwargs.pop("_adapter_model_path", None)

            if _adapter_model_path is None:
                _adapter_model_path = find_adapter_config_file(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    _commit_hash=commit_hash,
                    **adapter_kwargs,
                )
            if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
                with open(_adapter_model_path, "r", encoding="utf-8") as f:
                    _adapter_model_path = pretrained_model_name_or_path
                    pretrained_model_name_or_path = json.load(f)["base_model_name_or_path"]
        else:
            _adapter_model_path = None

        # change device_map into a map if we passed an int, a str or a torch.device
        if isinstance(device_map, torch.device):
            device_map = {"": device_map}
        elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            try:
                device_map = {"": torch.device(device_map)}
            except RuntimeError:
                raise ValueError(
                    "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                    f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
                )
        elif isinstance(device_map, int):
            if device_map < 0:
                raise ValueError(
                    "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
                )
            else:
                device_map = {"": device_map}

        if device_map is not None:
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
            elif not low_cpu_mem_usage:
                raise ValueError("Passing along a `device_map` requires `low_cpu_mem_usage=True`")

        if low_cpu_mem_usage:
            if is_deepspeed_zero3_enabled():
                raise ValueError(
                    "DeepSpeed Zero-3 is not compatible with `low_cpu_mem_usage=True` or with passing a `device_map`."
                )
            elif not is_accelerate_available():
                raise ImportError(
                    f"Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install 'accelerate>={ACCELERATE_MIN_VERSION}'`"
                )

        # handling bnb config from kwargs, remove after `load_in_{4/8}bit` deprecation.
        if load_in_4bit or load_in_8bit:
            if quantization_config is not None:
                raise ValueError(
                    "You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing "
                    "`quantization_config` argument at the same time."
                )

            # preparing BitsAndBytesConfig from kwargs
            config_dict = {k: v for k, v in kwargs.items() if k in inspect.signature(BitsAndBytesConfig).parameters}
            config_dict = {**config_dict, "load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit}
            quantization_config, kwargs = BitsAndBytesConfig.from_dict(
                config_dict=config_dict, return_unused_kwargs=True, **kwargs
            )
            logger.warning(
                "The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. "
                "Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead."
            )

        from_pt = not (from_tf | from_flax)

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            # In case one passes a config to `from_pretrained` + "attn_implementation"
            # override the `_attn_implementation` attribute to `attn_implementation` of the kwargs
            # Please see: https://github.com/huggingface/transformers/issues/28038

            # Overwrite `config._attn_implementation` by the one from the kwargs --> in auto-factory
            # we pop attn_implementation from the kwargs but this handles the case where users
            # passes manually the config to `from_pretrained`.
            config = copy.deepcopy(config)

            kwarg_attn_imp = kwargs.pop("attn_implementation", None)
            if kwarg_attn_imp is not None:
                config._attn_implementation = kwarg_attn_imp

            model_kwargs = kwargs

        pre_quantized = getattr(config, "quantization_config", None) is not None
        if pre_quantized or quantization_config is not None:
            if pre_quantized:
                config.quantization_config = AutoHfQuantizer.merge_quantization_configs(
                    config.quantization_config, quantization_config
                )
            else:
                config.quantization_config = quantization_config
            hf_quantizer = AutoHfQuantizer.from_config(config.quantization_config, pre_quantized=pre_quantized)
        else:
            hf_quantizer = None

        if hf_quantizer is not None:
            hf_quantizer.validate_environment(
                torch_dtype=torch_dtype, from_tf=from_tf, from_flax=from_flax, device_map=device_map
            )
            torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)
            device_map = hf_quantizer.update_device_map(device_map)

            # In order to ensure popular quantization methods are supported. Can be disable with `disable_telemetry`
            user_agent["quant"] = hf_quantizer.quantization_config.quant_method.value

            # Force-set to `True` for more mem efficiency
            if low_cpu_mem_usage is None:
                low_cpu_mem_usage = True
                logger.warning("`low_cpu_mem_usage` was None, now set to True since model is quantized.")
        is_quantized = hf_quantizer is not None

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False
        sharded_metadata = None
        # Load model
        loading_info = None

        # Keep in fp32 modules
        keep_in_fp32_modules = None
        use_keep_in_fp32_modules = False

        if gguf_file is not None and hf_quantizer is not None:
            raise ValueError(
                "You cannot combine Quantization and loading a model from a GGUF file, try again by making sure you did not passed a `quantization_config` or that you did not load a quantized model from the Hub."
            )

        if pretrained_model_name_or_path is not None and gguf_file is None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if is_local:
                if from_tf and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
                ):
                    # Load from a TF 1.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
                ):
                    # Load from a TF 2.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
                elif from_flax and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
                ):
                    # Load from a Flax checkpoint in priority if from_flax
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
                elif use_safetensors is not False and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
                ):
                    # Load from a safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant)
                    )
                elif use_safetensors is not False and os.path.isfile(
                    os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                    )
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                    )
                    is_sharded = True
                elif not use_safetensors and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant))
                ):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant)
                    )
                elif not use_safetensors and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
                ):
                    # Load from a sharded PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
                    )
                    is_sharded = True
                # At this stage we don't have a weight file so we will raise an error.
                elif not use_safetensors and (
                    os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index"))
                    or os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME))
                ):
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path} but there is a file for TensorFlow weights. Use"
                        " `from_tf=True` to load this model from those weights."
                    )
                elif not use_safetensors and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
                ):
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path} but there is a file for Flax weights. Use `from_flax=True`"
                        " to load this model from those weights."
                    )
                elif use_safetensors:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path}."
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)}, {_add_variant(SAFE_WEIGHTS_NAME, variant)},"
                        f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME + '.index'} or {FLAX_WEIGHTS_NAME} found in directory"
                        f" {pretrained_model_name_or_path}."
                    )
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path + ".index")):
                if not from_tf:
                    raise ValueError(
                        f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
                        "from_tf to True to load from this checkpoint."
                    )
                archive_file = os.path.join(subfolder, pretrained_model_name_or_path + ".index")
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(pretrained_model_name_or_path)
            else:
                # set correct filename
                if from_tf:
                    filename = TF2_WEIGHTS_NAME
                elif from_flax:
                    filename = FLAX_WEIGHTS_NAME
                elif use_safetensors is not False:
                    filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
                else:
                    filename = _add_variant(WEIGHTS_NAME, variant)

                try:
                    # Load from URL or cache if already cached
                    cached_file_kwargs = {
                        "cache_dir": cache_dir,
                        "force_download": force_download,
                        "proxies": proxies,
                        "resume_download": resume_download,
                        "local_files_only": local_files_only,
                        "token": token,
                        "user_agent": user_agent,
                        "revision": revision,
                        "subfolder": subfolder,
                        "_raise_exceptions_for_gated_repo": False,
                        "_raise_exceptions_for_missing_entries": False,
                        "_commit_hash": commit_hash,
                    }
                    resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

                    # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                    # result when internet is up, the repo and revision exist, but the file does not.
                    if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                        elif use_safetensors:
                            if revision == "main":
                                resolved_archive_file, revision, is_sharded = auto_conversion(
                                    pretrained_model_name_or_path, **cached_file_kwargs
                                )
                            cached_file_kwargs["revision"] = revision
                            if resolved_archive_file is None:
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} "
                                    "and thus cannot be loaded with `safetensors`. Please make sure that the model has "
                                    "been saved with `safe_serialization=True` or do not set `use_safetensors=True`."
                                )
                        else:
                            # This repo has no safetensors file of any kind, we switch to PyTorch.
                            filename = _add_variant(WEIGHTS_NAME, variant)
                            resolved_archive_file = cached_file(
                                pretrained_model_name_or_path, filename, **cached_file_kwargs
                            )
                    if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if not local_files_only and not is_offline_mode():
                        if resolved_archive_file is not None:
                            if filename in [WEIGHTS_NAME, WEIGHTS_INDEX_NAME]:
                                # If the PyTorch file was found, check if there is a safetensors file on the repository
                                # If there is no safetensors file on the repositories, start an auto conversion
                                safe_weights_name = SAFE_WEIGHTS_INDEX_NAME if is_sharded else SAFE_WEIGHTS_NAME
                                has_file_kwargs = {
                                    "revision": revision,
                                    "proxies": proxies,
                                    "token": token,
                                    "cache_dir": cache_dir,
                                    "local_files_only": local_files_only,
                                }
                                cached_file_kwargs = {
                                    "cache_dir": cache_dir,
                                    "force_download": force_download,
                                    "resume_download": resume_download,
                                    "local_files_only": local_files_only,
                                    "user_agent": user_agent,
                                    "subfolder": subfolder,
                                    "_raise_exceptions_for_gated_repo": False,
                                    "_raise_exceptions_for_missing_entries": False,
                                    "_commit_hash": commit_hash,
                                    **has_file_kwargs,
                                }
                                if not has_file(pretrained_model_name_or_path, safe_weights_name, **has_file_kwargs):
                                    Thread(
                                        target=auto_conversion,
                                        args=(pretrained_model_name_or_path,),
                                        kwargs={"ignore_errors_during_conversion": True, **cached_file_kwargs},
                                        name="Thread-autoconversion",
                                    ).start()
                        else:
                            # Otherwise, no PyTorch file was found, maybe there is a TF or Flax model file.
                            # We try those to give a helpful error message.
                            has_file_kwargs = {
                                "revision": revision,
                                "proxies": proxies,
                                "token": token,
                                "cache_dir": cache_dir,
                                "local_files_only": local_files_only,
                            }
                            if has_file(pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **has_file_kwargs):
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for TensorFlow weights."
                                    " Use `from_tf=True` to load this model from those weights."
                                )
                            elif has_file(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME, **has_file_kwargs):
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for Flax weights. Use"
                                    " `from_flax=True` to load this model from those weights."
                                )
                            elif variant is not None and has_file(
                                pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs
                            ):
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file without the variant"
                                    f" {variant}. Use `variant=None` to load this model from those weights."
                                )
                            else:
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)}, {_add_variant(SAFE_WEIGHTS_NAME, variant)},"
                                    f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                                )

                except EnvironmentError:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise
                except Exception as e:
                    # For any other exception, we throw a generic error.
                    raise EnvironmentError(
                        f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                        " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                        f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)},"
                        f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                    ) from e

            if is_local:
                logger.info(f"loading weights file {archive_file}")
                resolved_archive_file = archive_file
            else:
                logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")
        elif gguf_file:
            from .modeling_gguf_pytorch_utils import load_gguf_checkpoint

            # Case 1: the GGUF file is present locally
            if os.path.isfile(gguf_file):
                gguf_path = gguf_file
            # Case 2: The GGUF path is a location on the Hub
            # Load from URL or cache if already cached
            else:
                cached_file_kwargs = {
                    "cache_dir": cache_dir,
                    "force_download": force_download,
                    "proxies": proxies,
                    "resume_download": resume_download,
                    "local_files_only": local_files_only,
                    "token": token,
                    "user_agent": user_agent,
                    "revision": revision,
                    "subfolder": subfolder,
                    "_raise_exceptions_for_gated_repo": False,
                    "_raise_exceptions_for_missing_entries": False,
                    "_commit_hash": commit_hash,
                }

                gguf_path = cached_file(pretrained_model_name_or_path, gguf_file, **cached_file_kwargs)

            state_dict = load_gguf_checkpoint(gguf_path, return_tensors=True)["tensors"]

            resolved_archive_file = None
            is_sharded = False
        else:
            resolved_archive_file = None

        # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
        if is_sharded:
            # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                pretrained_model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=commit_hash,
            )

        if (
            is_safetensors_available()
            and isinstance(resolved_archive_file, str)
            and resolved_archive_file.endswith(".safetensors")
        ):
            with safe_open(resolved_archive_file, framework="pt") as f:
                metadata = f.metadata()

            if metadata.get("format") == "pt":
                pass
            elif metadata.get("format") == "tf":
                from_tf = True
                logger.info("A TensorFlow safetensors file is being loaded in a PyTorch model.")
            elif metadata.get("format") == "flax":
                from_flax = True
                logger.info("A Flax safetensors file is being loaded in a PyTorch model.")
            elif metadata.get("format") == "mlx":
                # This is a mlx file, we assume weights are compatible with pt
                pass
            else:
                raise ValueError(
                    f"Incompatible safetensors file. File metadata is not ['pt', 'tf', 'flax', 'mlx'] but {metadata.get('format')}"
                )

        from_pt = not (from_tf | from_flax)

        # load pt weights early so that we know which dtype to init the model under
        if from_pt:
            if not is_sharded and state_dict is None:
                # Time to load the checkpoint
                state_dict = load_state_dict(resolved_archive_file)

            # set dtype to instantiate the model under:
            # 1. If torch_dtype is not None, we use that dtype
            # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
            #    weights entry that is of a floating type - we assume all floating dtype weights are of the same dtype
            # we also may have config.torch_dtype available, but we won't rely on it till v5
            dtype_orig = None

            if torch_dtype is not None:
                if isinstance(torch_dtype, str):
                    if torch_dtype == "auto":
                        if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
                            torch_dtype = config.torch_dtype
                            logger.info(f"Will use torch_dtype={torch_dtype} as defined in model's config object")
                        else:
                            if is_sharded and "dtype" in sharded_metadata:
                                torch_dtype = sharded_metadata["dtype"]
                            elif not is_sharded:
                                torch_dtype = get_state_dict_dtype(state_dict)
                            else:
                                one_state_dict = load_state_dict(resolved_archive_file[0])
                                torch_dtype = get_state_dict_dtype(one_state_dict)
                                del one_state_dict  # free CPU memory
                            logger.info(
                                "Since the `torch_dtype` attribute can't be found in model's config object, "
                                "will use torch_dtype={torch_dtype} as derived from model's weights"
                            )
                    elif hasattr(torch, torch_dtype):
                        torch_dtype = getattr(torch, torch_dtype)
                    else:
                        raise ValueError(
                            f'`torch_dtype` can be one of: `torch.dtype`, `"auto"` or a string of a valid `torch.dtype`, but received {torch_dtype}'
                        )
                dtype_orig = cls._set_default_torch_dtype(torch_dtype)

            # Check if `_keep_in_fp32_modules` is not None
            use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (
                (torch_dtype == torch.float16) or hasattr(hf_quantizer, "use_keep_in_fp32_modules")
            )

            if is_sharded:
                loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
            else:
                loaded_state_dict_keys = list(state_dict.keys())

            if gguf_path is None and (low_cpu_mem_usage or (use_keep_in_fp32_modules and is_accelerate_available())):
                # In case some weights need to be kept in float32 and accelerate is not installed,
                # we later on want to take the path where state_dict is not None, that is the one
                # that do not require accelerate.
                state_dict = None

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        init_contexts = [no_init_weights(_enable=_fast_init)]

        if is_deepspeed_zero3_enabled() and not is_quantized:
            import deepspeed

            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            init_contexts = [deepspeed.zero.Init(config_dict_or_path=deepspeed_config())] + init_contexts
        elif low_cpu_mem_usage:
            init_contexts.append(init_empty_weights())

        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        config = cls._autoset_attn_implementation(
            config, use_flash_attention_2=use_flash_attention_2, torch_dtype=torch_dtype, device_map=device_map
        )

        with ContextManagers(init_contexts):
            # Let's make sure we don't run the init function of buffer modules
            model = cls(config, *model_args, **model_kwargs)

        # If we init with `zero3`, add an attr to the model so we can check downstream for issues
        model._transformers_zero3_init_used = is_deepspeed_zero3_enabled() and not is_quantized

        # make sure we use the model's config since the __init__ call might have copied it
        config = model.config

        # Check first if we are `from_pt`
        if use_keep_in_fp32_modules:
            if is_accelerate_available() and not is_deepspeed_zero3_enabled():
                low_cpu_mem_usage = True
            keep_in_fp32_modules = model._keep_in_fp32_modules
        else:
            keep_in_fp32_modules = []

        if hf_quantizer is not None:
            hf_quantizer.preprocess_model(
                model=model, device_map=device_map, keep_in_fp32_modules=keep_in_fp32_modules
            )

            # We store the original dtype for quantized models as we cannot easily retrieve it
            # once the weights have been quantized
            # Note that once you have loaded a quantized model, you can't change its dtype so this will
            # remain a single source of truth
            config._pre_quantization_dtype = torch_dtype

        if isinstance(device_map, str):
            special_dtypes = {}

            if hf_quantizer is not None:
                special_dtypes.update(hf_quantizer.get_special_dtypes_update(model, torch_dtype))

            special_dtypes.update(
                {
                    name: torch.float32
                    for name, _ in model.named_parameters()
                    if any(m in name for m in keep_in_fp32_modules)
                }
            )

            target_dtype = torch_dtype

            if hf_quantizer is not None:
                target_dtype = hf_quantizer.adjust_target_dtype(target_dtype)

            no_split_modules = model._get_no_split_modules(device_map)
            if device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
                raise ValueError(
                    "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                    "'sequential'."
                )

            device_map_kwargs = {"no_split_module_classes": no_split_modules}
            if "special_dtypes" in inspect.signature(infer_auto_device_map).parameters:
                device_map_kwargs["special_dtypes"] = special_dtypes
            elif len(special_dtypes) > 0:
                logger.warning(
                    "This model has some weights that should be kept in higher precision, you need to upgrade "
                    "`accelerate` to properly deal with them (`pip install --upgrade accelerate`)."
                )
            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    model,
                    dtype=target_dtype,
                    low_zero=(device_map == "balanced_low_0"),
                    max_memory=max_memory,
                    **device_map_kwargs,
                )
            else:
                max_memory = get_max_memory(max_memory)
            if hf_quantizer is not None:
                max_memory = hf_quantizer.adjust_max_memory(max_memory)
            device_map_kwargs["max_memory"] = max_memory

            # Make sure tied weights are tied before creating the device map.
            model.tie_weights()
            device_map = infer_auto_device_map(model, dtype=target_dtype, **device_map_kwargs)

            if hf_quantizer is not None:
                hf_quantizer.validate_environment(device_map=device_map)

        elif device_map is not None:
            model.tie_weights()
            tied_params = find_tied_parameters(model)
            # check if we don't have tied param in different devices
            check_tied_parameters_on_same_device(tied_params, device_map)

        if from_tf:
            if resolved_archive_file.endswith(".index"):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from .modeling_tf_pytorch_utils import load_tf2_checkpoint_in_pytorch_model

                    model, loading_info = load_tf2_checkpoint_in_pytorch_model(
                        model, resolved_archive_file, allow_missing_keys=True, output_loading_info=True
                    )
                except ImportError:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed."
                        " Please see https://pytorch.org/ and https://www.tensorflow.org/install/ for installation"
                        " instructions."
                    )
                    raise
        elif from_flax:
            try:
                from .modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model

                model = load_flax_checkpoint_in_pytorch_model(model, resolved_archive_file)
            except ImportError:
                logger.error(
                    "Loading a Flax model in PyTorch, requires both PyTorch and Flax to be installed. Please see"
                    " https://pytorch.org/ and https://flax.readthedocs.io/en/latest/installation.html for"
                    " installation instructions."
                )
                raise
        elif from_pt:
            # restore default dtype
            if dtype_orig is not None:
                torch.set_default_dtype(dtype_orig)

            (
                model,
                missing_keys,
                unexpected_keys,
                mismatched_keys,
                offload_index,
                error_msgs,
            ) = cls._load_pretrained_model(
                model,
                state_dict,
                loaded_state_dict_keys,  # XXX: rename?
                resolved_archive_file,
                pretrained_model_name_or_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                sharded_metadata=sharded_metadata,
                _fast_init=_fast_init,
                low_cpu_mem_usage=low_cpu_mem_usage,
                device_map=device_map,
                offload_folder=offload_folder,
                offload_state_dict=offload_state_dict,
                dtype=torch_dtype,
                hf_quantizer=hf_quantizer,
                keep_in_fp32_modules=keep_in_fp32_modules,
                gguf_path=gguf_path,
            )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        # If it is a model with generation capabilities, attempt to load the generation config
        if model.can_generate() and pretrained_model_name_or_path is not None:
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _from_auto=from_auto_class,
                    _from_pipeline=from_pipeline,
                    **kwargs,
                )
            except OSError:
                logger.info(
                    "Generation config file not found, using a generation config created from the model config."
                )
                pass

        # Dispatch model with hooks on all devices if necessary
        if device_map is not None:
            device_map_kwargs = {
                "device_map": device_map,
                "offload_dir": offload_folder,
                "offload_index": offload_index,
                "offload_buffers": offload_buffers,
            }
            if "skip_keys" in inspect.signature(dispatch_model).parameters:
                device_map_kwargs["skip_keys"] = model._skip_keys_device_placement
            # For HQQ method we force-set the hooks for single GPU envs
            if (
                "force_hooks" in inspect.signature(dispatch_model).parameters
                and hf_quantizer is not None
                and hf_quantizer.quantization_config.quant_method == QuantizationMethod.HQQ
            ):
                device_map_kwargs["force_hooks"] = True
            if (
                hf_quantizer is not None
                and hf_quantizer.quantization_config.quant_method == QuantizationMethod.FBGEMM_FP8
                and isinstance(device_map, dict)
                and ("cpu" in device_map.values() or "disk" in device_map.values())
            ):
                device_map_kwargs["offload_buffers"] = True

            if not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
                dispatch_model(model, **device_map_kwargs)

        if hf_quantizer is not None:
            hf_quantizer.postprocess_model(model)
            model.hf_quantizer = hf_quantizer

        if _adapter_model_path is not None:
            model.load_adapter(
                _adapter_model_path,
                adapter_name=adapter_name,
                token=token,
                adapter_kwargs=adapter_kwargs,
            )

        if output_loading_info:
            if loading_info is None:
                loading_info = {
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "mismatched_keys": mismatched_keys,
                    "error_msgs": error_msgs,
                }
            return model, loading_info

        return model
    
    # （47）加载预训练模型的权重到一个PyTorch模型实例中
    '''# 权重文件检查：根据device_map和offload_folder判断权重文件的存储方式，是否需要卸载到硬盘。
    # 模型权重绑定：在加载权重前，先对模型的权重进行绑定（如共享的嵌入层权重）。
    # 权重键处理：处理模型权重键的前缀，以匹配预训练模型的状态字典。
    # 权重尺寸不匹配处理：如果设置了ignore_mismatched_sizes，则忽略尺寸不匹配的权重。
    # 初始化权重：使用快速初始化或自定义初始化方法来初始化模型权重。
    # 权重加载：根据state_dict和resolved_archive_file加载模型权重。
    # 权重卸载：如果设置了low_cpu_mem_usage，将部分权重卸载到硬盘以节省CPU内存。
    # 量化处理：如果模型是量化的，使用量化器对权重进行处理。
    # 权重尺寸不匹配警告：如果存在尺寸不匹配的权重，则发出警告。
    # 未使用和缺失权重处理：报告未使用的权重和需要新初始化的缺失权重。'''
    
    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        loaded_keys,
        resolved_archive_file,
        pretrained_model_name_or_path,
        ignore_mismatched_sizes=False,
        sharded_metadata=None,
        _fast_init=True,
        low_cpu_mem_usage=False,
        device_map=None,
        offload_folder=None,
        offload_state_dict=None,
        dtype=None,
        hf_quantizer=None,
        keep_in_fp32_modules=None,
        gguf_path=None,
    ):
        is_safetensors = False
        is_quantized = hf_quantizer is not None
        state_dict_folder = None
        state_dict_index = None

        if device_map is not None and "disk" in device_map.values():
            archive_file = (
                resolved_archive_file[0] if isinstance(resolved_archive_file, (list, tuple)) else resolved_archive_file
            )
            is_safetensors = archive_file.endswith(".safetensors")
            if offload_folder is None and not is_safetensors:
                raise ValueError(
                    "The current `device_map` had weights offloaded to the disk. Please provide an `offload_folder`"
                    " for them. Alternatively, make sure you have `safetensors` installed if the model you are using"
                    " offers the weights in this format."
                )
            if offload_folder is not None:
                os.makedirs(offload_folder, exist_ok=True)
            if offload_state_dict is None:
                offload_state_dict = True

        is_sharded_safetensors = is_safetensors and sharded_metadata is not None

        # tie the model weights before retrieving the state_dict
        model.tie_weights()

        # Retrieve missing & unexpected_keys
        model_state_dict = model.state_dict()
        expected_keys = list(model_state_dict.keys())
        prefix = model.base_model_prefix

        def _fix_key(key):
            if "beta" in key:
                return key.replace("beta", "bias")
            if "gamma" in key:
                return key.replace("gamma", "weight")
            return key

        original_loaded_keys = loaded_keys
        loaded_keys = [_fix_key(key) for key in loaded_keys]

        if len(prefix) > 0:
            has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
        else:
            has_prefix_module = False
            expects_prefix_module = False

        # key re-naming operations are never done on the keys
        # that are loaded, but always on the keys of the newly initialized model
        remove_prefix_from_model = not has_prefix_module and expects_prefix_module
        add_prefix_to_model = has_prefix_module and not expects_prefix_module

        if remove_prefix_from_model:
            _prefix = f"{prefix}."
            expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(_prefix)]
            expected_keys = [s[len(_prefix) :] if s.startswith(_prefix) else s for s in expected_keys]
        elif add_prefix_to_model:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]

        missing_keys = sorted(set(expected_keys) - set(loaded_keys))
        unexpected_keys = set(loaded_keys) - set(expected_keys)

        # Remove nonpersistent buffers from unexpected keys: they are not in the state dict but will be in the model
        # buffers
        model_buffers = {n for n, _ in model.named_buffers()}
        if remove_prefix_from_model:
            model_buffers = {key[len(_prefix) :] if key.startswith(_prefix) else key for key in model_buffers}
        elif add_prefix_to_model:
            model_buffers = {".".join([prefix, key]) for key in model_buffers}
        unexpected_keys = sorted(unexpected_keys - model_buffers)

        model.tie_weights()
        if device_map is None and not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
            ptrs = collections.defaultdict(list)
            for name, tensor in model.state_dict().items():
                id_tensor = id_tensor_storage(tensor)
                ptrs[id_tensor].append(name)

            # These are all the pointers of shared tensors.
            tied_params = [names for _, names in ptrs.items() if len(names) > 1]
        else:
            # id function doesn't work for meta tensor so we need this function
            tied_params = find_tied_parameters(model)

        for group in tied_params:
            if remove_prefix_from_model:
                group = [key[len(_prefix) :] if key.startswith(_prefix) else key for key in group]
            elif add_prefix_to_model:
                group = [".".join([prefix, key]) for key in group]
            missing_in_group = [k for k in missing_keys if k in group]
            if len(missing_in_group) > 0 and len(missing_in_group) < len(group):
                missing_keys = [k for k in missing_keys if k not in missing_in_group]

        # Some models may have keys that are not in the state by design, removing them before needlessly warning
        # the user.
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
        if hf_quantizer is not None:
            missing_keys = hf_quantizer.update_missing_keys(model, missing_keys, prefix)

        # retrieve weights on meta device and put them back on CPU.
        # This is not ideal in terms of memory, but if we don't do that not, we can't initialize them in the next step
        if low_cpu_mem_usage:
            for key in missing_keys:
                if key in list(model_state_dict.keys()):
                    key = key
                elif f"{prefix}.{key}" in list(model_state_dict.keys()):
                    key = f"{prefix}.{key}"
                elif key.startswith(prefix) and ".".join(key.split(".")[1:]) in list(model_state_dict.keys()):
                    key = ".".join(key.split(".")[1:])
                param = model_state_dict[key]

                # upcast in fp32 if any
                target_dtype = dtype
                if (
                    keep_in_fp32_modules is not None
                    and dtype == torch.float16
                    and any(
                        module_to_keep_in_fp32 in key.split(".") for module_to_keep_in_fp32 in keep_in_fp32_modules
                    )
                ):
                    target_dtype = torch.float32

                if param.device == torch.device("meta"):
                    value = torch.empty(*param.size(), dtype=target_dtype)
                    if (
                        not is_quantized
                        or getattr(hf_quantizer, "requires_parameters_quantization", False)
                        or not hf_quantizer.check_quantized_param(
                            model, param_value=value, param_name=key, state_dict={}
                        )
                    ):
                        set_module_tensor_to_device(model, key, "cpu", value)
                    else:
                        hf_quantizer.create_quantized_param(model, value, key, "cpu", state_dict, unexpected_keys)

        # retrieve uninitialized modules and initialize before maybe overriding that with the pretrained weights.
        if _fast_init:
            if not ignore_mismatched_sizes:
                if remove_prefix_from_model:
                    _loaded_keys = [f"{prefix}.{k}" for k in loaded_keys]
                elif add_prefix_to_model:
                    _loaded_keys = [k[len(prefix) + 1 :] for k in loaded_keys]
                else:
                    _loaded_keys = loaded_keys
                not_initialized_submodules = set_initialized_submodules(model, _loaded_keys)
                # If we're about to tie the output embeds to the input embeds we don't need to init them
                if hasattr(model.config, "tie_word_embeddings") and model.config.tie_word_embeddings:
                    output_embeddings = model.get_output_embeddings()
                    if output_embeddings is not None:
                        # Still need to initialize if there is a bias term since biases are not tied.
                        if not hasattr(output_embeddings, "bias") or output_embeddings.bias is None:
                            output_embeddings._is_hf_initialized = True
            else:
                not_initialized_submodules = dict(model.named_modules())
            # This will only initialize submodules that are not marked as initialized by the line above.
            if is_deepspeed_zero3_enabled() and not is_quantized:
                import deepspeed

                not_initialized_parameters = list(
                    set(
                        itertools.chain.from_iterable(
                            submodule.parameters(recurse=False) for submodule in not_initialized_submodules.values()
                        )
                    )
                )
                with deepspeed.zero.GatheredParameters(not_initialized_parameters, modifier_rank=0):
                    model.apply(model._initialize_weights)
            else:
                model.apply(model._initialize_weights)

        # Set some modules to fp32 if any
        if keep_in_fp32_modules is not None:
            for name, param in model.named_parameters():
                if any(module_to_keep_in_fp32 in name.split(".") for module_to_keep_in_fp32 in keep_in_fp32_modules):
                    # param = param.to(torch.float32) does not work here as only in the local scope.
                    param.data = param.data.to(torch.float32)

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if len(cls.base_model_prefix) > 0 and not hasattr(model, cls.base_model_prefix) and has_prefix_module:
            start_prefix = cls.base_model_prefix + "."
        if len(cls.base_model_prefix) > 0 and hasattr(model, cls.base_model_prefix) and not has_prefix_module:
            model_to_load = getattr(model, cls.base_model_prefix)
            base_model_expected_keys = list(model_to_load.state_dict().keys())
            if any(key in expected_keys_not_prefixed and key not in base_model_expected_keys for key in loaded_keys):
                raise ValueError(
                    "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
                    "properly saved?"
                )
            if device_map is not None:
                device_map = {k.replace(f"{cls.base_model_prefix}.", ""): v for k, v in device_map.items()}

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    # If the checkpoint is sharded, we may not have the key here.
                    if checkpoint_key not in state_dict:
                        continue
                    model_key = checkpoint_key
                    if remove_prefix_from_model:
                        # The model key starts with `prefix` but `checkpoint_key` doesn't so we add it.
                        model_key = f"{prefix}.{checkpoint_key}"
                    elif add_prefix_to_model:
                        # The model key doesn't start with `prefix` but `checkpoint_key` does so we remove it.
                        model_key = ".".join(checkpoint_key.split(".")[1:])

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        if (
                            state_dict[checkpoint_key].shape[-1] == 1
                            and state_dict[checkpoint_key].numel() * 2 == model_state_dict[model_key].numel()
                        ):
                            # This skips size mismatches for 4-bit weights. Two 4-bit values share an 8-bit container, causing size differences.
                            # Without matching with module type or paramter type it seems like a practical way to detect valid 4bit weights.
                            pass
                        else:
                            mismatched_keys.append(
                                (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                            )
                            del state_dict[checkpoint_key]
            return mismatched_keys

        if resolved_archive_file is not None:
            folder = os.path.sep.join(resolved_archive_file[0].split(os.path.sep)[:-1])
        else:
            folder = None
        if device_map is not None and is_safetensors:
            param_device_map = expand_device_map(device_map, original_loaded_keys, start_prefix)
            str_dtype = str(dtype).replace("torch.", "") if dtype is not None else "float32"
            if sharded_metadata is None:
                archive_file = (
                    resolved_archive_file[0]
                    if isinstance(resolved_archive_file, (list, tuple))
                    else resolved_archive_file
                )
                weight_map = {p: archive_file for p in original_loaded_keys}
            else:
                weight_map = {p: os.path.join(folder, f) for p, f in sharded_metadata["weight_map"].items()}
            offload_index = {
                p[len(start_prefix) :]: {"safetensors_file": f, "weight_name": p, "dtype": str_dtype}
                for p, f in weight_map.items()
                if p.startswith(start_prefix) and param_device_map[p[len(start_prefix) :]] == "disk"
            }
        else:
            offload_index = None

        if state_dict is not None:
            # Whole checkpoint
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                add_prefix_to_model,
                remove_prefix_from_model,
                ignore_mismatched_sizes,
            )

            # For GGUF models `state_dict` is never set to None as the state dict is always small
            if gguf_path:
                error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
                    model_to_load,
                    state_dict,
                    loaded_keys,
                    start_prefix,
                    expected_keys,
                    device_map=device_map,
                    offload_folder=offload_folder,
                    offload_index=offload_index,
                    state_dict_folder=state_dict_folder,
                    state_dict_index=state_dict_index,
                    dtype=dtype,
                    hf_quantizer=hf_quantizer,
                    is_safetensors=is_safetensors,
                    keep_in_fp32_modules=keep_in_fp32_modules,
                    unexpected_keys=unexpected_keys,
                )
            else:
                # Sharded checkpoint or whole but low_cpu_mem_usage==True
                assign_to_params_buffers = check_support_param_buffer_assignment(
                    model_to_load, state_dict, start_prefix
                )
                error_msgs = _load_state_dict_into_model(
                    model_to_load, state_dict, start_prefix, assign_to_params_buffers
                )

        else:
            # This should always be a list but, just to be sure.
            if not isinstance(resolved_archive_file, list):
                resolved_archive_file = [resolved_archive_file]

            error_msgs = []
            mismatched_keys = []
            if not is_safetensors:
                offload_index = {} if device_map is not None and "disk" in device_map.values() else None
            if offload_state_dict:
                state_dict_folder = tempfile.mkdtemp()
                state_dict_index = {}
            else:
                state_dict_folder = None
                state_dict_index = None

            if is_sharded_safetensors:
                disk_only_shard_files = get_disk_only_shard_files(
                    device_map, sharded_metadata=sharded_metadata, start_prefix=start_prefix
                )
                disk_only_shard_files = [os.path.join(folder, f) for f in disk_only_shard_files]
            else:
                disk_only_shard_files = []

            if len(resolved_archive_file) > 1:
                resolved_archive_file = logging.tqdm(resolved_archive_file, desc="Loading checkpoint shards")
            assign_to_params_buffers = None
            for shard_file in resolved_archive_file:
                # Skip the load for shards that only contain disk-offloaded weights when using safetensors for the offload.
                if shard_file in disk_only_shard_files:
                    continue
                state_dict = load_state_dict(shard_file, is_quantized=is_quantized)

                # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
                # matching the weights in the model.
                mismatched_keys += _find_mismatched_keys(
                    state_dict,
                    model_state_dict,
                    original_loaded_keys,
                    add_prefix_to_model,
                    remove_prefix_from_model,
                    ignore_mismatched_sizes,
                )
                if low_cpu_mem_usage:
                    if is_fsdp_enabled() and not is_local_dist_rank_0() and not is_quantized:
                        for key, param in model_to_load.state_dict().items():
                            if param.device == torch.device("meta"):
                                set_module_tensor_to_device(
                                    model_to_load, key, "cpu", torch.empty(*param.size(), dtype=dtype)
                                )
                    else:
                        new_error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
                            model_to_load,
                            state_dict,
                            loaded_keys,
                            start_prefix,
                            expected_keys,
                            device_map=device_map,
                            offload_folder=offload_folder,
                            offload_index=offload_index,
                            state_dict_folder=state_dict_folder,
                            state_dict_index=state_dict_index,
                            dtype=dtype,
                            hf_quantizer=hf_quantizer,
                            is_safetensors=is_safetensors,
                            keep_in_fp32_modules=keep_in_fp32_modules,
                            unexpected_keys=unexpected_keys,
                        )
                        error_msgs += new_error_msgs
                else:
                    # Sharded checkpoint or whole but low_cpu_mem_usage==True
                    if assign_to_params_buffers is None:
                        assign_to_params_buffers = check_support_param_buffer_assignment(
                            model_to_load, state_dict, start_prefix
                        )
                    error_msgs += _load_state_dict_into_model(
                        model_to_load, state_dict, start_prefix, assign_to_params_buffers
                    )

                # force memory release
                del state_dict
                gc.collect()

            if offload_index is not None and len(offload_index) > 0:
                if model != model_to_load:
                    # We need to add the prefix of the base model
                    prefix = cls.base_model_prefix
                    if not is_safetensors:
                        for weight_name in offload_index:
                            shutil.move(
                                os.path.join(offload_folder, f"{weight_name}.dat"),
                                os.path.join(offload_folder, f"{prefix}.{weight_name}.dat"),
                            )
                    offload_index = {f"{prefix}.{key}": value for key, value in offload_index.items()}
                if not is_safetensors:
                    save_offload_index(offload_index, offload_folder)
                    offload_index = None

            if offload_state_dict:
                # Load back temporarily offloaded state dict
                load_offloaded_weights(model_to_load, state_dict_index, state_dict_folder)
                shutil.rmtree(state_dict_folder)

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            archs = [] if model.config.architectures is None else model.config.architectures
            warner = logger.warning if model.__class__.__name__ in archs else logger.info
            warner(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, offload_index, error_msgs
    # （48）从模型中检索与特定名称列表相关的模块。
    def retrieve_modules_from_names(self, names, add_prefix=False, remove_prefix=False):
        module_keys = {".".join(key.split(".")[:-1]) for key in names}

        # torch.nn.ParameterList is a special case where two parameter keywords
        # are appended to the module name, *e.g.* bert.special_embeddings.0
        module_keys = module_keys.union(
            {".".join(key.split(".")[:-2]) for key in names if len(key) > 0 and key[-1].isdigit()}
        )

        retrieved_modules = []
        # retrieve all modules that has at least one missing weight name
        for name, module in self.named_modules():
            if remove_prefix:
                _prefix = f"{self.base_model_prefix}."
                name = name[len(_prefix) :] if name.startswith(_prefix) else name
            elif add_prefix:
                name = ".".join([self.base_model_prefix, name]) if len(name) > 0 else self.base_model_prefix

            if name in module_keys:
                retrieved_modules.append(module)

        return retrieved_modules
    # （49）以较低的内存消耗加载预训练的模型。这个函数是实验性的，并且有特定的使用条件和步骤。
    @staticmethod
    def _load_pretrained_model_low_mem(
        model, loaded_state_dict_keys, resolved_archive_file, start_prefix="", hf_quantizer=None
    ):
        """
        This is an experimental function that loads the model using ~1.x model size CPU memory

        Before you call it do:

        1. save which state_dict keys are available
        2. drop state_dict before model is created, since the latter takes 1x model size memory

        Here then we continue:

        3. switch to the meta device all params/buffers that are going to be replaced from the loaded state_dict
        4. load state_dict 2nd time
        5. replace the params/buffers from the state_dict

        Currently, it doesn't handle missing_keys, unexpected_keys, mismatched_keys. It can't handle deepspeed. To
        handle bitsandbytes, needs non-empty hf_quantizer argument.
        """

        _move_model_to_meta(model, loaded_state_dict_keys, start_prefix)
        state_dict = load_state_dict(resolved_archive_file)
        expected_keys = loaded_state_dict_keys  # plug for missing expected_keys. TODO: replace with proper keys
        error_msgs = _load_state_dict_into_meta_model(
            model,
            state_dict,
            loaded_state_dict_keys,
            start_prefix,
            expected_keys=expected_keys,
            hf_quantizer=hf_quantizer,
        )
        return error_msgs
    # （50）将自定义模型类注册到一个特定的自动模型类（auto class）。这个功能在机器学习库 transformers 中很常见，特别是当你想要使用库中提供的自动模型加载功能时。
    @classmethod
    def register_for_auto_class(cls, auto_class="AutoModel"):
        """
        Register this class with a given auto class. This should only be used for custom models as the ones in the
        library are already mapped with an auto class.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoModel"`):
                The auto class to register this new model with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class
    # （51）将一个模型转换为使用PyTorch的原生注意力实现，
    def to_bettertransformer(self) -> "PreTrainedModel":
        """
        Converts the model to use [PyTorch's native attention
        implementation](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html), integrated to
        Transformers through [Optimum library](https://huggingface.co/docs/optimum/bettertransformer/overview). Only a
        subset of all Transformers models are supported.

        PyTorch's attention fastpath allows to speed up inference through kernel fusions and the use of [nested
        tensors](https://pytorch.org/docs/stable/nested.html). Detailed benchmarks can be found in [this blog
        post](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2).

        Returns:
            [`PreTrainedModel`]: The model converted to BetterTransformer.
        """
        if not is_optimum_available():
            raise ImportError("The package `optimum` is required to use Better Transformer.")

        from optimum.version import __version__ as optimum_version

        if version.parse(optimum_version) < version.parse("1.7.0"):
            raise ImportError(
                f"Please install optimum>=1.7.0 to use Better Transformer. The version {optimum_version} was found."
            )

        from optimum.bettertransformer import BetterTransformer

        return BetterTransformer.transform(self)
    # （52）是撤销之前通过to_bettertransformer方法应用的转换，以便恢复到原始的模型状态，这通常用于保存模型。
    def reverse_bettertransformer(self):
        """
        Reverts the transformation from [`~PreTrainedModel.to_bettertransformer`] so that the original modeling is
        used, for example in order to save the model.

        Returns:
            [`PreTrainedModel`]: The model converted back to the original modeling.
        """
        if not is_optimum_available():
            raise ImportError("The package `optimum` is required to use Better Transformer.")

        from optimum.version import __version__ as optimum_version

        if version.parse(optimum_version) < version.parse("1.7.0"):
            raise ImportError(
                f"Please install optimum>=1.7.0 to use Better Transformer. The version {optimum_version} was found."
            )

        from optimum.bettertransformer import BetterTransformer

        return BetterTransformer.reverse(self)
    # （53）输入数据是否含有填充（padding）并提示用户使用注意力掩码（attention mask）
    def warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask):
        """
        Shows a one-time warning if the input_ids appear to contain padding and no attention mask was given.
        """

        # Skip the check during tracing.
        if is_torch_fx_proxy(input_ids) or torch.jit.is_tracing() or is_torchdynamo_compiling():
            return

        if (attention_mask is not None) or (self.config.pad_token_id is None):
            return

        # Check only the first and last input IDs to reduce overhead.
        if self.config.pad_token_id in input_ids[:, [-1, 0]]:
            warn_string = (
                "We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See "
                "https://huggingface.co/docs/transformers/troubleshooting"
                "#incorrect-output-when-padding-tokens-arent-masked."
            )

            # If the pad token is equal to either BOS, EOS, or SEP, we do not know whether the user should use an
            # attention_mask or not. In this case, we should still show a warning because this is a rare case.
            if (
                (self.config.bos_token_id is not None and self.config.bos_token_id == self.config.pad_token_id)
                or (self.config.eos_token_id is not None and self.config.eos_token_id == self.config.pad_token_id)
                or (self.config.sep_token_id is not None and self.config.sep_token_id == self.config.pad_token_id)
            ):
                warn_string += (
                    f"\nYou may ignore this warning if your `pad_token_id` ({self.config.pad_token_id}) is identical "
                    f"to the `bos_token_id` ({self.config.bos_token_id}), `eos_token_id` ({self.config.eos_token_id}), "
                    f"or the `sep_token_id` ({self.config.sep_token_id}), and your input is not padded."
                )

            logger.warning_once(warn_string)
    # （54）检查量化训练是否被启用
    @property
    def _is_quantized_training_enabled(self):
        warnings.warn(
            "`_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead",
            FutureWarning,
        )

        if not hasattr(self, "hf_quantizer"):
            return False

        return self.hf_quantizer.is_trainable

# 元编程，它在修改一个名为 PreTrainedModel 的类的 push_to_hub 方法
PreTrainedModel.push_to_hub = copy_func(PreTrainedModel.push_to_hub)    # 获取 PreTrainedModel 类的 push_to_hub 方法的引用，然后使用 copy_func 函数复制这个方法。接着，它将复制的方法重新赋值给 PreTrainedModel.push_to_hub。这样做的目的可能是为了在不改变原始方法定义的情况下，对方法进行一些额外的操作或修改。
if PreTrainedModel.push_to_hub.__doc__ is not None:                     # 检查 push_to_hub 方法是否有文档字符串（__doc__ 属性）。如果存在文档字符串，这行代码将使用 str.format 方法来格式化它
    PreTrainedModel.push_to_hub.__doc__ = PreTrainedModel.push_to_hub.__doc__.format(
        object="model", object_class="AutoModel", object_files="model file"
    )

# （1）这个类继承自 nn.Module，用于计算SQuAD任务中起始位置的对数几率（logits）。它接收模型的隐藏状态作为输入，并通过一个线性层输出起始位置的得分。
class PoolerStartLogits(nn.Module):
    """
    这个模块接收模型的隐藏状态作为输入，并输出一个表示起始位置的对数几率的张量。
    SQuAD（斯坦福问答任务，Stanford Question Answering Dataset）是一个广泛使用的机器阅读理解数据集，由斯坦福大学的研究者在2016年创建。它是为了评估计算机理解自然语言的能力而设计的，特别是它们阅读和理解给定文本段落后回答问题的能力。
    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model.
    """
    # 类构造函数创建一个dense层：接收一个配置对象 config，它通常是 PretrainedConfig 的一个实例，包含了模型的配置信息。hidden_size 属性将被用来设置密集层的输入维度。
    def __init__(self, config: PretrainedConfig):
        super().__init__()                              
        self.dense = nn.Linear(config.hidden_size, 1)   # 将输入的隐藏状态从 hidden_size 维度转换到1维。这个1维输出将代表每个位置作为起始位置的对数几率。
    
    def forward(
        self, hidden_states: torch.FloatTensor, p_mask: Optional[torch.FloatTensor] = None              
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                模型最终的隐藏状态，形状为 (batch_size, seq_len, hidden_size)
            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                # p_mask可选参数，是一个掩码张量，用于标记序列中无效位置的标记，比如查询符号和特殊符号（如PAD, SEP, CLS）。如果 p_mask 为1，则表示对应的token应该被屏蔽。
        Returns:
            `torch.FloatTensor`: The start logits for SQuAD.
        计算逻辑：首先，hidden_states 通过 self.dense 进行线性变换。.squeeze(-1): 将结果张量的最后一个维度压缩，从 (batch_size, seq_len, 1) 变为 (batch_size, seq_len)。
        """
        x = self.dense(hidden_states).squeeze(-1)

        if p_mask is not None:
            if get_parameter_dtype(self) == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x

# （2）用于计算SQuAD任务中结束位置的对数几率。它使用另一个线性层，并且可以选择性地接收起始位置的状态作为输入，以便更准确地预测结束位置。
class PoolerEndLogits(nn.Module):
    """
    Compute SQuAD end logits from sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps`
            to use.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        p_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                The hidden states of the first tokens for the labeled span.
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                The position of the first token for the labeled span.
            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        <Tip>

        One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides
        `start_states`.

        </Tip>

        Returns:
            `torch.FloatTensor`: The end logits for SQuAD.
        """
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if get_parameter_dtype(self) == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x

# （3）这个类用于SQuAD 2.0任务，预测问题是否有可能的答案。它通过结合起始位置的状态和CLS令牌的状态，输出一个表示答案存在可能性的分数。
class PoolerAnswerClass(nn.Module):
    """
    根据classification计算SQuAD 2.0答案类，并启动令牌隐藏状态start tokens hidden states。
    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        cls_index: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                The hidden states of the first tokens for the labeled span.
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                The position of the first token for the labeled span.
            cls_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Position of the CLS token for each sentence in the batch. If `None`, takes the last token.

        <Tip>

        One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides
        `start_states`.

        </Tip>

        Returns:
            `torch.FloatTensor`: The SQuAD 2.0 answer class.
        """
        # No dependency on end_feature so that we can obtain one single `cls_logits` for each sample.
        hsz = hidden_states.shape[-1]
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions).squeeze(-2)  # shape (bsz, hsz)

        if cls_index is not None:
            cls_index = cls_index[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            cls_token_state = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, -1, :]  # shape (bsz, hsz)

        x = self.dense_0(torch.cat([start_states, cls_token_state], dim=-1))
        x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)

        return x

# （4）使用 dataclass 装饰的类，定义了SQuAD模型输出的数据结构。它包含损失值和各种top概率及索引，这些信息在模型训练和推理过程中非常有用。
@dataclass
class SquadHeadOutput(ModelOutput):
    """
    使用[`~modeling_utils.SQuADHead`]输出问答模型的基类。
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided):
            Classification loss分类损失= start token分类损失和 end token (and is_impossible if provided) 分类损失之和
        start_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            记录top config.start_n_top开始令牌可能性的概率（波束搜索）。
        start_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            顶部config.start_n_top开始令牌可能性的索引（波束搜索）。
        end_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            记录顶部`config.start_n_top*config.end_n_top`结束令牌可能性的概率（波束搜索）。
        end_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            顶部`config.start_n_top*config.end_n_top`结束令牌可能性的索引（波束搜索）。
        cls_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or `end_positions` is not provided):
            记录答案的“不可能”标签的概率。
    """

    loss: Optional[torch.FloatTensor] = None
    start_top_log_probs: Optional[torch.FloatTensor] = None
    start_top_index: Optional[torch.LongTensor] = None
    end_top_log_probs: Optional[torch.FloatTensor] = None
    end_top_index: Optional[torch.LongTensor] = None
    cls_logits: Optional[torch.FloatTensor] = None

# （5）这个类继承自 nn.Module，是SQuAD模型的“头部”，负责整合起始位置、结束位置和答案类别的预测。它使用 PoolerStartLogits、PoolerEndLogits 和 PoolerAnswerClass 类来执行这些任务。
class SQuADHead(nn.Module):
    r"""
    A SQuAD head inspired by XLNet.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps`
            to use.
    """

    def __init__(self, config):
        super().__init__()
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top

        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

    @replace_return_docstrings(output_type=SquadHeadOutput, config_class=PretrainedConfig)
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        cls_index: Optional[torch.LongTensor] = None,
        is_impossible: Optional[torch.LongTensor] = None,
        p_mask: Optional[torch.FloatTensor] = None,
        return_dict: bool = False,
    ) -> Union[SquadHeadOutput, Tuple[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                Final hidden states of the model on the sequence tokens.
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Positions of the first token for the labeled span.
            end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Positions of the last token for the labeled span.
            cls_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Position of the CLS token for each sentence in the batch. If `None`, takes the last token.
            is_impossible (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Whether the question has a possible answer in the paragraph or not.
            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
        """
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if cls_index is not None and is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)

                # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
                total_loss += cls_loss * 0.5

            return SquadHeadOutput(loss=total_loss) if return_dict else (total_loss,)

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = nn.functional.softmax(start_logits, dim=-1)  # shape (bsz, slen)

            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = nn.functional.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_states = torch.einsum("blh,bl->bh", hidden_states, start_log_probs)
            cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)

            if not return_dict:
                return (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits)
            else:
                return SquadHeadOutput(
                    start_top_log_probs=start_top_log_probs,
                    start_top_index=start_top_index,
                    end_top_log_probs=end_top_log_probs,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

# （6）这个类用于从序列的隐藏状态中生成单一向量摘要。它可以根据配置选择不同的方法来提取摘要，例如取最后一个token的状态、第一个token的状态、所有token的平均状态，或者使用特定的分类token位置。
class SequenceSummary(nn.Module):
    r"""
    Compute a single vector summary of a sequence hidden states.

    Args:
        config ([`PretrainedConfig`]):
            The config used by the model. Relevant arguments in the config class of the model are (refer to the actual
            config class of your model for the default values it uses):

            - **summary_type** (`str`) -- The method to use to make this summary. Accepted values are:

                - `"last"` -- Take the last token hidden state (like XLNet)
                - `"first"` -- Take the first token hidden state (like Bert)
                - `"mean"` -- Take the mean of all tokens hidden states
                - `"cls_index"` -- Supply a Tensor of classification token position (GPT/GPT-2)
                - `"attn"` -- Not implemented now, use multi-head attention

            - **summary_use_proj** (`bool`) -- Add a projection after the vector extraction.
            - **summary_proj_to_labels** (`bool`) -- If `True`, the projection outputs to `config.num_labels` classes
              (otherwise to `config.hidden_size`).
            - **summary_activation** (`Optional[str]`) -- Set to `"tanh"` to add a tanh activation to the output,
              another string or `None` will add no activation.
            - **summary_first_dropout** (`float`) -- Optional dropout probability before the projection and activation.
            - **summary_last_dropout** (`float`)-- Optional dropout probability after the projection and activation.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()

        self.summary_type = getattr(config, "summary_type", "last")
        if self.summary_type == "attn":
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError

        self.summary = Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        activation_string = getattr(config, "summary_activation", None)
        self.activation: Callable = get_activation(activation_string) if activation_string else Identity()

        self.first_dropout = Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

    def forward(
        self, hidden_states: torch.FloatTensor, cls_index: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        """
        Compute a single vector summary of a sequence hidden states.

        Args:
            hidden_states (`torch.FloatTensor` of shape `[batch_size, seq_len, hidden_size]`):
                The hidden states of the last layer.
            cls_index (`torch.LongTensor` of shape `[batch_size]` or `[batch_size, ...]` where ... are optional leading dimensions of `hidden_states`, *optional*):
                Used if `summary_type == "cls_index"` and takes the last token of the sequence as classification token.

        Returns:
            `torch.FloatTensor`: The summary of the sequence hidden states.
        """
        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = hidden_states.mean(dim=1)
        elif self.summary_type == "cls_index":
            if cls_index is None:
                cls_index = torch.full_like(
                    hidden_states[..., :1, :],
                    hidden_states.shape[-2] - 1,
                    dtype=torch.long,
                )
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),))
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = hidden_states.gather(-2, cls_index).squeeze(-2)  # shape (bsz, XX, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output

# （1）用于递归地从潜在的容器（如分布式训练中使用的）中解包模型。它确保无论模型被包装了多少层，都能获取到最内层的实际模型。
def unwrap_model(model: nn.Module, recursive: bool = False) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
        recursive (`bool`, *optional*, defaults to `False`):
            Whether to recursively extract all cases of `module.module` from `model` as well as unwrap child sublayers
            recursively, not just the top-level distributed containers.
    """
    # Use accelerate implementation if available (should always be the case when using torch)
    # This is for pytorch, as we also have to handle things like dynamo
    if is_accelerate_available():
        kwargs = {}
        if recursive:
            if not is_accelerate_available("0.29.0"):
                raise RuntimeError(
                    "Setting `recursive=True` to `unwrap_model` requires `accelerate` v0.29.0. Please upgrade your version of accelerate"
                )
            else:
                kwargs["recursive"] = recursive
        return extract_model_from_parallel(model, **kwargs)
    else:
        # since there could be multiple levels of wrapping, unwrap recursively
        if hasattr(model, "module"):
            return unwrap_model(model.module)
        else:
            return model

# （2）用于扩展设备映射，将参数名映射到它们应该被放置的设备上。它处理了模型参数和设备之间的对应关系。
def expand_device_map(device_map, param_names, start_prefix):
    """
    Expand a device map to return the correspondance parameter name to device.
    """
    new_device_map = {}
    param_names = [p[len(start_prefix) :] for p in param_names if p.startswith(start_prefix)]
    for module, device in device_map.items():
        new_device_map.update(
            {p: device for p in param_names if p == module or p.startswith(f"{module}.") or module == ""}
        )
    return new_device_map

# （3）返回只包含被卸载到磁盘上的权重的分片文件列表。它用于管理和优化模型的权重存储，特别是在模型太大不适合完全加载到内存中时。
def get_disk_only_shard_files(device_map, sharded_metadata, start_prefix):
    """
    Returns the list of shard files containing only weights offloaded to disk.
    """

    weight_map = {
        p[len(start_prefix) :]: v for p, v in sharded_metadata["weight_map"].items() if p.startswith(start_prefix)
    }
    files_content = collections.defaultdict(list)
    for weight_name, filename in weight_map.items():
        while len(weight_name) > 0 and weight_name not in device_map:
            weight_name = ".".join(weight_name.split(".")[:-1])
        files_content[filename].append(device_map[weight_name])

    return [fname for fname, devices in files_content.items() if set(devices) == {"disk"}]
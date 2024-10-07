# 概述
torch Version: 2.4.0

# 项目结构
pwd：miniconda3/envs/Omni/lib/python3.12/site-packages/torch/nn
tree -L 1
.
├── attention
├── backends
├── common_types.py
├── cpp.py
├── functional.py
├── functional.pyi
├── grad.py
├── __init__.py
├── init.py
├── intrinsic
├── modules
├── parallel
├── **parameter.py**        ：定义_ParameterMeta、Parameter、UninitializedTensorMixin、UninitializedParameter、UninitializedBuffer五个类
├── **parameter.pyi**       ：parameter的类型提示文件，允许开发者指定函数、方法、类及其成员的预期类型，从而提高代码的可读性和可维护性，同时帮助静态类型检查器捕获潜在的类型错误。
├── __pycache__
├── qat
├── quantizable
├── quantized
├── _reduction.py
└── utils



## nn：eural network神经网络模块


## xpu：设备无关的代码，含有torch.Tensor等
xpu 通常是指扩展处理单元（Extended Processing Unit）。这是一个泛化的术语，可能用来代表不同类型的计算设备，比如 GPU（图形处理单元）、CPU（中央处理单元）和其他自定义硬件加速器（如 FPGA、TPU，甚至是未来的定制 AI 加速器）。xpu 通常用于开发中表示设备无关的代码，帮助简化和通用化在不同硬件设备之间的计算。
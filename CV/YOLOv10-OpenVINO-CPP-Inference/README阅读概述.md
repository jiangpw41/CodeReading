# 项目介绍
基于英特尔的OpenVINO（开放视觉推理与神经优化）工具包的，使用C++进行YOLOv10高效准确实时推理的项目。
## 支持的模型格式
- 模型权重格式：支持跨平台的 `ONNX` 和 `OpenVINO IR` 
- 模型权重精度：支持`FP32`, `FP16` and `INT8`
- 模型权重形状：支持动态加载模型权重shape
## 支持的OS
Ubuntu `18.04`, `20.04`, `22.04`.

## 系统包依赖
| Dependency | Version  |
| ---------- | -------- |
| OpenVINO   | >=2023.3 |
| OpenCV     | >=3.2.0  |
| C++        | >=14     |
| CMake      | >=3.10.2 |

# 项目结构
```bash
├── assets                  # 文件夹：五张用于测试的图片
├── Dockerfile              # 文件：用于从本地构建docker镜像
├── notebooks               # 文件夹：一个粗糙的格式转换notebook
│   └── YOLOv10_exporter.ipynb
├── README.md               # 操作指南
└── src                     # 文件夹：build和inference的C++文件，本质是一个Inference+三个main+一个utils
    ├── CMakeLists.txt
    ├── inference.cc        # .cc是C++的文件后缀，和.cpp没有本质区别，负责实现
    ├── inference.h         # .cc对应的.h文件，负责声明
    ├── main_camera.cc
    ├── main_detect.cc
    ├── main_video.cc
    ├── utils.cc
    └── utils.h
```
编译时，在src目录下

```bash
mkdir build
cd build
cmake ..      # 命令告诉CMake在当前目录的父目录中查找CMakeLists.txt文件，并根据该文件中的指令生成构建系统。
make
```
cmake是一个跨平台的自动化构建系统，它使用CMakeLists.txt文件（位于当前目录的上一级，即..所指向的目录）来定义项目的构建过程。详情见该文件注释。
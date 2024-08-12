# 概述
2017年Google提出Transformer架构，2019年Huggingface团队推出transformers库。
本目录下的transformers文件夹来自，github仓库中./src文件夹。
下载日期为2024.8.12，Latest Release version为v4.44.0

'''
<git clone git@github.com:huggingface/transformers.git>
'''
官方中文文档：https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hans.md

# 项目结构
HF 提供的基础模型类有 PreTrainedModel, TFPreTrainedModel, and FlaxPreTrainedModel，在根目录下看名字就可识别

# 已阅读
## modeling_utils.py
transformers.modeling_utils，总行数为5156行，定义了26个函数，8个类（其中6个特定于SQuAD任务），核心是：
- PreTrainedModel类（3390行，54个方法，20个属性）
- ModuleUtilsMixin类（308行，14个方法），是所有预训练模型的基类

# 正在阅读
无

# 待阅读
## modeling_outputs.py
1753行
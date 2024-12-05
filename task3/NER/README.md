# W2NER

本代码参考 W2NER论文及模型编写，论文链接[AAAI Press Formatting Instructions for Authors Using LaTeX -- A Guide (arxiv.org)](https://arxiv.org/pdf/2112.10070.pdf)



## 代码运行

在colab上运行main.py即可(或许上传空文件夹时colab会忽略，需手动创建，比如output、saved_models等)



## 代码结构

由于本人并不是一开始就使用了W2NER模型，各个模型之间的代码也是有所不同的，我将代码分为utils、Model、Config、Trainer、main，只需重写utils内部的APIs和Model(或许有部分Trainer)即可迁移使用不同的模型。

### utils

分为common和DataProcess以及对不同文本的接口APIs

#### common

提供logger、read_from_file、write_to_file函数

#### DataProcess

提供Process类，是数据预处理的接口，分为encode预处理和decode格式化

#### APIs

需重写APIDataset、api_encode、api_decode，分别对应不同的接口，其中api_encode接收原始json数据，返回APIDataset所接收的数据；APIDataset接收api_encode的数据，返回Model所需的Dataset；api_decode接收Model输出的output，返回格式化的字典(json)数据

### Model

模型主体

### Config

通用配置

### Trainer

训练类，分为train、eval、predict


# 可用的模型

直接使用的模型都放在 [**cnstd-cnocr-models**](https://huggingface.co/breezedeus/cnstd-cnocr-models) 项目中，可免费下载使用。如果下载太慢，也可以从 [百度云盘](https://pan.baidu.com/s/1wvIjbjw95akE-h_o1HQd9w?pwd=nocr) 下载， 提取码为 ` nocr`。具体方法可参考 [使用方法](usage.md) 。

模型分为两大类，1）来自 **[CnSTD](https://github.com/breezedeus/cnstd)** 的**检测模型**；2）来自 CnOCR 的**识别模型**。



## 检测模型

具体说明请参考 **[CnSTD 文档](https://github.com/breezedeus/cnstd)**，以下仅罗列出可用模型：

| `det_model_name`                                             | PyTorch 版本 | ONNX 版本 | 模型原始来源 | 模型文件大小 | 支持语言                       | 是否支持竖排文字识别 |
| ------------------------------------------------------------ | ------------ | --------- | ------------ | ------------ | ------------------------------ | -------------------- |
| db_shufflenet_v2                                             | √            | X         | cnocr        | 18 M         | 简体中文、繁体中文、英文、数字 | √                    |
| **db_shufflenet_v2_small**                                   | √            | X         | cnocr        | 12 M         | 简体中文、繁体中文、英文、数字 | √                    |
| [db_shufflenet_v2_tiny](https://mp.weixin.qq.com/s/fHPNoGyo72EFApVhEgR6Nw) | √            | X         | cnocr        | 7.5 M        | 简体中文、繁体中文、英文、数字 | √                    |
| db_mobilenet_v3                                              | √            | X         | cnocr        | 16 M         | 简体中文、繁体中文、英文、数字 | √                    |
| db_mobilenet_v3_small                                        | √            | X         | cnocr        | 7.9 M        | 简体中文、繁体中文、英文、数字 | √                    |
| db_resnet34                                                  | √            | X         | cnocr        | 86 M         | 简体中文、繁体中文、英文、数字 | √                    |
| db_resnet18                                                  | √            | X         | cnocr        | 47 M         | 简体中文、繁体中文、英文、数字 | √                    |
| ch_PP-OCRv3_det                                              | X            | √         | ppocr        | 2.3 M        | 简体中文、繁体中文、英文、数字 | √                    |
| ch_PP-OCRv2_det                                              | X            | √         | ppocr        | 2.2 M        | 简体中文、繁体中文、英文、数字 | √                    |
| **en_PP-OCRv3_det**                                          | X            | √         | ppocr        | 2.3 M        | **英文**、数字                 | √                    |


> **Note**
>
> 列 **`PyTorch 版本`** 为 `√` 表示此模型支持 `det_model_backend=='pytorch'`；列 **`ONNX 版本`** 为 `√` 表示此模型支持 `det_model_backend=='onnx'`；取值为 `X` 则表示不支持对应的取值。

## 识别模型

CnOCR 自 **V2.1.2** 之后，可直接使用的识别模型包含两类：1）CnOCR 自己训练的模型，通常会包含 PyTorch 和 ONNX 版本；2）从其他ocr引擎搬运过来的训练好的外部模型，ONNX化后用于 CnOCR 中。



### 1) CnOCR 自己训练的模型

CnOCR 自己训练的模型都支持**常见简体中文、英文和数字**的识别，大家也可以基于这些模型在自己的领域数据上继续精调模型。模型列表如下：

| `rec_model_name`        | PyTorch 版本 | ONNX 版本 | 参数规模 | 模型文件大小 | 准确度 | 平均推断耗时（毫秒/图） |
| ----------------------- | ------------ | --------- | -------- | ------------ | ------ | ----------------------- |
| densenet\_lite\_114-fc  | √            | √         | 1.3 M    | 4.9 M        | 0.9274 | 9.229                   |
| densenet\_lite\_124-fc  | √            | √         | 1.3 M    | 5.1 M        | 0.9429 | 10.112                  |
| densenet\_lite\_134-fc  | √            | √         | 1.4 M    | 5.4 M        | 0.954  | 10.843                  |
| densenet\_lite\_136-fc  | √            | √         | 1.5M     | 5.9 M        | 0.9631 | 11.499                  |
| densenet\_lite\_134-gru | √            | X         | 2.9 M    | 11 M         | 0.9738 | 17.042                  |
| densenet\_lite\_136-gru | √            | X         | 3.1 M    | 12 M         | 0.9756 | 17.725                  |

一些说明：

1. 模型名称是由**局部编码**模型和**序列编码**模型名称拼接而成，以符合"-"分割。
2. 列 **`PyTorch 版本`** 为 `√` 表示此模型支持 `model_backend=='pytorch'`；列 **`ONNX 版本`** 为 `√` 表示此模型支持 `model_backend=='onnx'`；取值为 `X` 则表示不支持对应的取值。
3. `平均耗时` 是针对 `PyTorch 版本` 获得的，**`ONNX 版本` 耗时大致是 `PyTorch 版本` 的一半。**

CnOCR 的自有模型从结构上可以分为两阶段：第一阶段是获得ocr图片的局部编码向量，第二部分是对局部编码向量进行序列学习，获得序列编码向量。目前的PyTorch版本的两个阶段分别包含以下模型：

1. 局部编码模型（emb model）
   - **`densenet_lite_<numbers>`**：一个微型的`densenet`网络；其中的`<number>`表示模型中每个block包含的层数。
   - **`densenet`**：一个小型的`densenet`网络；
2. 序列编码模型（seq model）
   - **`fc`**：两层的全连接网络；
   - **`gru`**：一层的GRU网络；
   - **`lstm`**：一层的LSTM网络。

### 2) 外部模型

以下模型是 [**PaddleOCR**](https://github.com/PaddlePaddle/PaddleOCR) 中模型的 **ONNX** 版本，所以不会依赖 **PaddlePaddle** 相关工具包，故而也不支持基于这些模型在自己的领域数据上继续精调模型。这些模型应该都支持**竖排文字**。

| `model_name`          | PyTorch 版本 | ONNX 版本 | 支持语言                 | 是否支持竖排文字识别 | 模型文件大小 |
| --------------------- | ------------ | --------- | ------------------------ | -------------------- | ------------ |
| ch_PP-OCRv3           | X            | √         | 简体中文、英文、数字     | √                    | 10 M         |
| ch_ppocr_mobile_v2.0  | X            | √         | 简体中文、英文、数字     | √                    | 4.2 M        |
| en_PP-OCRv3           | X            | √         | **英文**、数字           | √                    | 8.5 M        |
| en_number_mobile_v2.0 | X            | √         | **英文**、数字           | √                    | 1.8 M        |
| chinese_cht_PP-OCRv3  | X            | √         | **繁体中文**、英文、数字 | X                    | 11 M         |

更多模型可参考 [PaddleOCR/models_list.md](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.5/doc/doc_ch/models_list.md) 。如有其他外语（如日、韩等）识别需求，可在 **知识星球** [**CnOCR/CnSTD私享群**](https://t.zsxq.com/FEYZRJQ) 中向作者提出建议。

## 可直接使用的模型

cnocr的ocr模型可以分为两阶段：第一阶段是获得ocr图片的局部编码向量，第二部分是对局部编码向量进行序列学习，获得序列编码向量。目前的PyTorch版本的两个阶段分别包含以下模型：

1. 局部编码模型（emb model）
   * **`densenet_lite_<numbers>`**：一个微型的`densenet`网络；其中的`<number>`表示模型中每个block包含的层数。
   * **`densenet`**：一个小型的`densenet`网络；
2. 序列编码模型（seq model）
   * **`fc`**：两层的全连接网络；
   * **`gru`**：一层的GRU网络；
   * **`lstm`**：一层的LSTM网络。

cnocr **V2.1** 目前包含以下可直接使用的模型，训练好的模型都放在 **[cnstd-cnocr-models](https://github.com/breezedeus/cnstd-cnocr-models)** 项目中，可免费下载使用：

| Name                    | PyTorch 版本 | ONNX 版本 | 参数规模  | 模型文件大小 | 准确度    | 平均推断耗时（毫秒/图） |
| ----------------------- | ---------- | ------- | ----- | ------ | ------ | ------------ |
| densenet\_lite\_114-fc  | √          | √       | 1.3 M | 4.9 M  | 0.9274 | 9.229        |
| densenet\_lite\_124-fc  | √          | √       | 1.3 M | 5.1 M  | 0.9429 | 10.112       |
| densenet\_lite\_134-fc  | √          | √       | 1.4 M | 5.4 M  | 0.954  | 10.843       |
| densenet\_lite\_136-fc  | √          | √       | 1.5M  | 5.9 M  | 0.9631 | 11.499       |
| densenet\_lite\_134-gru | √          | X       | 2.9 M | 11 M   | 0.9738 | 17.042       |
| densenet\_lite\_136-gru | √          | X       | 3.1 M | 12 M   | 0.9756 | 17.725       |

一些说明：

1. 模型名称是由**局部编码**模型和**序列编码**模型名称拼接而成，以符合"-"分割。
2. 列 **`PyTorch 版本`** 为 `√` 表示此模型支持 `model_backend=='pytorch'`；列 **`ONNX 版本`** 为 `√` 表示此模型支持 `model_backend=='onnx'`；取值为 `X` 则表示不支持对应的取值。
3. `平均耗时` 是针对 `PyTorch 版本` 获得的，**`ONNX 版本` 耗时大致是 `PyTorch 版本` 的一半。**

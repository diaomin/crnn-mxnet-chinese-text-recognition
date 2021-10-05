## 可直接使用的模型

cnocr的ocr模型可以分为两阶段：第一阶段是获得ocr图片的局部编码向量，第二部分是对局部编码向量进行序列学习，获得序列编码向量。目前的PyTorch版本的两个阶段分别包含以下模型：

1. 局部编码模型（emb model）
    * **`densenet-s`**：一个小型的`densenet`网络；
2. 序列编码模型（seq model）
    * **`lstm`**：一层的LSTM网络；
    * **`gru`**：一层的GRU网络；
    * **`fc`**：两层的全连接网络。



cnocr **V2.0** 目前包含以下可直接使用的模型，训练好的模型都放在 **[cnstd-cnocr-models](https://github.com/breezedeus/cnstd-cnocr-models)** 项目中，可免费下载使用：

| 模型名称 | 局部编码模型 | 序列编码模型 | 模型大小 | 迭代次数 | 测试集准确率  |
| :------- | ------------ | ------------ | -------- | ------ | -------- |
| densenet-s-gru | densenet-s | gru | 11 M | 11 | 95.5% |
| densenet-s-fc | densenet-s | fc | 8.7 M | 39 | 91.9% |


> 模型名称是由局部编码模型和序列编码模型名称拼接而成。





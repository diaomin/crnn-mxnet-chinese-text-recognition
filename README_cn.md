# cnocr
**cnocr**是用来做中文OCR的**Python 3**包。cnocr自带了训练好的识别模型，所以安装后即可直接使用。

目前使用的识别模型是**crnn**，识别准确度约为 `98.7%`。

本项目起源于我们自己 ([爱因互动 Ein+](https://einplus.cn)) 内部的项目需求，所以非常感谢公司的支持。


## 特色

本项目的大部分代码都fork自 [crnn-mxnet-chinese-text-recognition](https://github.com/diaomin/crnn-mxnet-chinese-text-recognition)，感谢作者。

但源项目使用起来不够方便，所以我在此基础上做了一些封装和重构。主要变化如下：

* 不再使用需要额外安装的MXNet WarpCTC Loss，改用原生的 MXNet CTC Loss。所以安装极简！

* 自带训练好的中文OCR识别模型。不再需要额外训练！

* 增加了预测（或推断）接口。所以使用方便！


## 安装

```bash
pip install cnocr
```

> 注意：请使用Python3 (3.4, 3.5, 3.6以及之后版本应该都行)，没测过Python2下是否ok。



## 使用方法

### 预测

以图片文件 [examples/rand_cn1.png](./examples/rand_cn1.png)为例，文件内容如下：

![examples/rand_cn1.png](/Users/king/Documents/WhatIHaveDone/Test/cnocr/examples/rand_cn1.png)



#### 代码引用

```python
from cnocr import CnOcr
ocr = CnOcr()
res = ocr.ocr_for_single_line('examples/rand_cn1.png')
print("Predicted Chars:", res)
```



首次使用cnocr时，系统会自动从[Dropbox](https://www.dropbox.com/s/5n09nxf4x95jprk/cnocr-models-v0.1.0.zip) 下载zip格式的模型压缩文件，并存于 `~/.cnocr`目录。

下载后的zip文件代码会自动对其解压，然后把解压后的模型相关文件放于`~/.cnocr/models`目录。

如果系统不能自动从[Dropbox](https://www.dropbox.com/s/5n09nxf4x95jprk/cnocr-models-v0.1.0.zip) 成功下载zip文件，则需要手动下载此zip文件并把它放于 `~/.cnocr`目录。

另一个下载地址是[百度云盘](https://pan.baidu.com/s/1s91985r0YBGbk_1cqgHa1Q) (提取码为`pg26`)。

放置好zip文件后，后面的事代码就会自动执行了。



上面预测代码段的返回结果如下：

```bash
Predicted Chars: ['笠', '淡', '嘿', '骅', '谧', '鼎', '皋', '姚', '歼', '蠢', '驼', '耳', '胬', '挝', '涯', '狗', '蒽', '子', '犷']
```



#### 脚本引用

也可以使用脚本模式预测：

```bash
python scripts/cnocr_predict.py --file examples/rand_cn1.png
```
返回结果和前面相同：
```bash
Predicted Chars: ['笠', '淡', '嘿', '骅', '谧', '鼎', '皋', '姚', '歼', '蠢', '驼', '耳', '胬', '挝', '涯', '狗', '蒽', '子', '犷']
```



### 训练自己的模型

cnocr安装后即可直接使用，但如果你**非要**训练自己的模型，请参考下面命令：

```bash
python scripts/cnocr_train.py --cpu 2 --num_proc 4 --loss ctc --dataset cn_ocr
```



## 未来工作

* 支持`空格`识别
* 修bugs（目前代码还比较凌乱。。）
* 完善测试用例
* 考虑使用MxNet的命令式编程重写代码，提升灵活性
* 尝试新模型，如 DenseNet、ResNet，进一步提升识别准确率


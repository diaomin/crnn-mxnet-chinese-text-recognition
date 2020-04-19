# Release Notes

### Update 2020.04.20: 发布 cnocr V1.1.0



### Update 2019.07.25: 发布 cnocr V1.0.0

`cnocr`发布了预测效率更高的新版本v1.0.0。**新版本的模型跟以前版本的模型不兼容**。所以如果大家是升级的话，需要重新下载最新的模型文件。具体说明见下面（流程和原来相同）。



主要改动如下：

-  **crnn模型支持可变长预测，提升预测效率**
-  支持利用特定数据对现有模型进行精调（继续训练）
-  修复bugs，如训练时`accuracy`一直为`0`
-  依赖的 `mxnet` 版本从`1.3.1`更新至 `1.4.1`

# cnocr

**cnocr**是用来做中文OCR的**Python 3**包。cnocr自带了训练好的识别模型，所以安装后即可直接使用。

本项目起源于我们自己 ([爱因互动 Ein+](https://einplus.cn)) 内部的项目需求，所以非常感谢公司的支持。

## 可直接使用的模型

cnocr的ocr模型可以分为两阶段：第一阶段是获得ocr图片的局部编码向量，第二部分是对局部编码向量进行序列学习，获得序列编码向量。目前两个阶段分别包含以下的模型：

1. 局部编码模型（emb model）
   * `conv`：多层的卷积网络；
   * `conv-lite`：更小的多层卷积网络；
   * `densenet`：一个小型的`densenet`网络；
   * `densenet-lite`：一个更小的`densenet`网络。
2. 序列编码模型（seq model）
   * `lstm`：两层的LSTM网络；
   * `gru`：两层的GRU网络；
   * `fc`：两层的全连接网络。



cnocr目前包含以下可直接使用的模型：

| 模型名称 | 局部编码模型 | 序列编码模型 | 模型大小 | 迭代次数 | 测试集准确率 |
| :------- | ------------ | ------------ | -------- | ------ | -------- |
| conv-lstm | conv | lstm | 36M | 50 | 98.5% |
| conv-lite-lstm | conv-lite | lstm | 23M | 45 | 98.6% |
| conv-lite-fc | conv-lite | fc | 20M | 27 | 98.6% |
| densenet-lite-lstm | densenet-lite | lstm | 8.6M | 42 | 98.6% |
| densenet-lite-fc | densenet-lite | fc | 6.8M | 32 | 97% |

> 模型名称是由局部编码模型和序列编码模型名称拼接而成。

## 特色

本项目的初期代码都fork自 [crnn-mxnet-chinese-text-recognition](https://github.com/diaomin/crnn-mxnet-chinese-text-recognition)，感谢作者。

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

首次使用cnocr时，系统会自动从[Dropbox](https://www.dropbox.com/s/7w8l3mk4pvkt34w/cnocr-models-v1.0.0.zip?dl=0)下载zip格式的模型压缩文件，并存于 `~/.cnocr`目录。
下载后的zip文件代码会自动对其解压，然后把解压后的模型相关文件放于`~/.cnocr/models`目录。

如果系统不能自动从[Dropbox](https://www.dropbox.com/s/7w8l3mk4pvkt34w/cnocr-models-v1.0.0.zip?dl=0)成功下载zip文件，则需要手动下载此zip文件并把它放于 `~/.cnocr`目录。
另一个下载地址是[百度云盘](https://pan.baidu.com/s/1DWV3H2UWmzOU6d48UbTYVw)(提取码为`ss81`)。
放置好zip文件后，后面的事代码就会自动执行了。



### 代码预测

主要包含三个函数，下面分别说明。



#### 1. 函数`CnOcr.ocr(img_fp)`

函数`CnOcr.ocr(img_fp)`可以对包含多行文字（或单行）的图片进行文字识别。



**函数说明**：

- 输入参数 `img_fp`: 可以是需要识别的图片文件路径（如上例）；或者是已经从图片文件中读入的数组，类型可以为`mx.nd.NDArray` 或  `np.ndarray`，取值应该是`[0，255]`的整数，维数应该是`(height, width, 3)`，第三个维度是channel，它应该是`RGB`格式的。
- 返回值：为一个嵌套的`list`，类似这样`[['第', '一', '行'], ['第', '二', '行'], ['第', '三', '行']]`。



**调用示例**：


```python
from cnocr import CnOcr
ocr = CnOcr()
res = ocr.ocr('examples/multi-line_cn1.png')
print("Predicted Chars:", res)
```

或：
```python
import mxnet as mx
from cnocr import CnOcr
ocr = CnOcr()
img_fp = 'examples/multi-line_cn1.png'
img = mx.image.imread(img_fp, 1)
res = ocr.ocr(img)
print("Predicted Chars:", res)
```



上面使用的图片文件 [examples/multi-line_cn1.png](./examples/multi-line_cn1.png)内容如下：

![examples/multi-line_cn1.png](./examples/multi-line_cn1.png)



上面预测代码段的返回结果如下：

```bash
Predicted Chars: [['网', '络', '支', '付', '并', '无', '本', '质', '的', '区', '别', '，', '因', '为'],
                  ['每', '一', '个', '手', '机', '号', '码', '和', '邮', '件', '地', '址', '背', '后'],
                  ['都', '会', '对', '应', '着', '一', '个', '账', '户', '一', '―', '这', '个', '账'],
                  ['户', '可', '以', '是', '信', '用', '卡', '账', '户', '、', '借', '记', '卡', '账'],
                  ['户', '，', '也', '包', '括', '邮', '局', '汇', '款', '、', '手', '机', '代'],
                  ['收', '、', '电', '话', '代', '收', '、', '预', '付', '费', '卡', '和', '点', '卡'],
                  ['等', '多', '种', '形', '式', '。']]
```



#### 2. 函数`CnOcr.ocr_for_single_line(img_fp)`

如果明确知道要预测的图片中只包含了单行文字，可以使用函数`CnOcr.ocr_for_single_line(img_fp)`进行识别。和 `CnOcr.ocr()`相比，`CnOcr.ocr_for_single_line()`结果可靠性更强，因为它不需要做额外的分行处理。

**函数说明**：

- 输入参数 `img_fp`: 可以是需要识别的单行文字图片文件路径（如上例）；或者是已经从图片文件中读入的数组，类型可以为`mx.nd.NDArray` 或  `np.ndarray`，取值应该是`[0，255]`的整数，维数应该是`(height, width)`或`(height, width, channel)`。如果没有channel，表示传入的就是灰度图片。第三个维度channel可以是`1`（灰度图片）或者`3`（彩色图片）。如果是彩色图片，它应该是`RGB`格式的。
- 返回值：为一个`list`，类似这样`['你', '好']`。



**调用示例**：

```python
from cnocr import CnOcr
ocr = CnOcr()
res = ocr.ocr_for_single_line('examples/rand_cn1.png')
print("Predicted Chars:", res)
```

或：

```python
import mxnet as mx
from cnocr import CnOcr
ocr = CnOcr()
img_fp = 'examples/rand_cn1.png'
img = mx.image.imread(img_fp, 1)
res = ocr.ocr_for_single_line(img)
print("Predicted Chars:", res)
```


对图片文件 [examples/rand_cn1.png](./examples/rand_cn1.png)：

![examples/rand_cn1.png](./examples/rand_cn1.png)

的预测结果如下：

```bash
Predicted Chars: ['笠', '淡', '嘿', '骅', '谧', '鼎', '臭', '姚', '歼', '蠢', '驼', '耳', '裔', '挝', '涯', '狗', '蒽', '子', '犷'] 
```



#### 3. 函数`CnOcr.ocr_for_single_lines(img_list)`

函数`CnOcr.ocr_for_single_lines(img_list)`可以**对多个单行文字图片进行批量预测**。函数`CnOcr.ocr(img_fp)`和`CnOcr.ocr_for_single_line(img_fp)`内部其实都是调用的函数`CnOcr.ocr_for_single_lines(img_list)`。



**函数说明**：

- 输入参数` img_list`: 为一个`list`；其中每个元素是已经从图片文件中读入的数组，类型可以为`mx.nd.NDArray` 或  `np.ndarray`，取值应该是`[0，255]`的整数，维数应该是`(height, width)`或`(height, width, channel)`。如果没有channel，表示传入的就是灰度图片。第三个维度channel可以是`1`（灰度图片）或者`3`（彩色图片）。如果是彩色图片，它应该是`RGB`格式的。
- 返回值：为一个嵌套的`list`，类似这样`[['第', '一', '行'], ['第', '二', '行'], ['第', '三', '行']]`。



**调用示例**：

```python
import mxnet as mx
from cnocr import CnOcr
ocr = CnOcr()
img_fp = 'examples/multi-line_cn1.png'
img = mx.image.imread(img_fp, 1).asnumpy()
line_imgs = line_split(img, blank=True)
line_img_list = [line_img for line_img, _ in line_imgs]
res = ocr.ocr_for_single_lines(line_img_list)
print("Predicted Chars:", res)
```



更详细的使用方法，可参考[tests/test_cnocr.py](./tests/test_cnocr.py)中提供的测试用例。



### 脚本引用

也可以使用脚本模式预测：

```bash
python scripts/cnocr_predict.py --file examples/multi-line_cn1.png
```

返回结果同上面。



### 训练自己的模型

cnocr安装后即可直接使用，但如果你**非要**训练自己的模型，请参考下面命令：

```bash
python scripts/cnocr_train.py --cpu 2 --num_proc 4 --loss ctc --dataset cn_ocr
```



现在也支持从已有模型利用特定数据精调模型，请参考下面命令：

```bash
python scripts/cnocr_train.py --cpu 2 --num_proc 4 --loss ctc --dataset cn_ocr --load_epoch 20
```



更多可参考脚本[scripts/run_cnocr_train.sh](./scripts/run_cnocr_train.sh)中的命令。



## 未来工作

* [x] 支持图片包含多行文字 (`Done`)
* [x] crnn模型支持可变长预测，提升灵活性 (`Done`)
* [x] 完善测试用例 (`Doing`)
* [x] 修bugs（目前代码还比较凌乱。。） (`Doing`)
* [ ] 支持`空格`识别（`V1.0.0`在训练集中加入了空格，但从预测结果看，空格依旧是识别不出来）
* 尝试新模型，如 DenseNet、ResNet，进一步提升识别准确率


# 使用方法

## 模型文件自动下载

首次使用cnocr时，系统会**自动下载** zip格式的模型压缩文件，并存于 `~/.cnocr`目录（Windows下默认路径为 `C:\Users\<username>\AppData\Roaming\cnocr`）。
下载后的zip文件代码会自动对其解压，然后把解压后的模型相关目录放于`~/.cnocr/2.1`目录中。

如果系统无法自动成功下载zip文件，则需要手动从 **[cnstd-cnocr-models](https://huggingface.co/breezedeus/cnstd-cnocr-models/tree/main)** 下载此zip文件并把它放于 `~/.cnocr/2.1`目录。如果下载太慢，也可以从 [百度云盘](https://pan.baidu.com/s/1N6HoYearUzU0U8NTL3K35A) 下载， 提取码为 ` gcig`。

放置好zip文件后，后面的事代码就会自动执行了。

## 预测代码

### 针对多行文字的图片识别

如果待识别的图片包含多行文字，或者可能包含多行文字（如下图），可以使用 `CnOcr.ocr()` 进行识别。

![多行文字图片](examples/multi-line_cn1.png)

**调用示例**：

```python
from cnocr import CnOcr

ocr = CnOcr()
res = ocr.ocr('docs/examples/multi-line_cn1.png')
print("Predicted Chars:", res)
```

或：

```python
from cnocr.utils import read_img
from cnocr import CnOcr

ocr = CnOcr()
img_fp = 'docs/examples/multi-line_cn1.png'
img = read_img(img_fp)
res = ocr.ocr(img)
print("Predicted Chars:", res)
```

### 针对单行文字的图片识别

如果明确知道待识别的图片包含单行文字（如下图），可以使用 `CnOcr.ocr_for_single_line()` 进行识别。

![单行文字图片](examples/helloworld.jpg)

**调用示例**：

```python
from cnocr import CnOcr

ocr = CnOcr()
res = ocr.ocr_for_single_line('docs/examples/helloworld.jpg')
print("Predicted Chars:", res)
```

或：

```python
from cnocr.utils import read_img
from cnocr import CnOcr

ocr = CnOcr()
img_fp = 'docs/examples/helloworld.jpg'
img = read_img(img_fp)
res = ocr.ocr_for_single_line(img)
print("Predicted Chars:", res)
```

## 效果示例

| 图片                                                                      | OCR结果                                                                                                                                                                                                                                                                                                                                                                                       |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![examples/helloworld.jpg](./examples/helloworld.jpg)                   | Hello world!你好世界                                                                                                                                                                                                                                                                                                                                                                            |
| ![examples/chn-00199989.jpg](./examples/chn-00199989.jpg)               | 铑泡胭释邑疫反隽寥缔                                                                                                                                                                                                                                                                                                                                                                                  |
| ![examples/chn-00199980.jpg](./examples/chn-00199980.jpg)               | 拇箬遭才柄腾戮胖惬炫                                                                                                                                                                                                                                                                                                                                                                                  |
| ![examples/chn-00199984.jpg](./examples/chn-00199984.jpg)               | 寿猿嗅髓孢刀谎弓供捣                                                                                                                                                                                                                                                                                                                                                                                  |
| ![examples/chn-00199985.jpg](./examples/chn-00199985.jpg)               | 马靼蘑熨距额猬要藕萼                                                                                                                                                                                                                                                                                                                                                                                  |
| ![examples/chn-00199981.jpg](./examples/chn-00199981.jpg)               | 掉江悟厉励.谌查门蠕坑                                                                                                                                                                                                                                                                                                                                                                                 |
| ![examples/00199975.jpg](./examples/00199975.jpg)                       | nd-chips fructed ast                                                                                                                                                                                                                                                                                                                                                                        |
| ![examples/00199978.jpg](./examples/00199978.jpg)                       | zouna unpayably Raqu                                                                                                                                                                                                                                                                                                                                                                        |
| ![examples/00199979.jpg](./examples/00199979.jpg)                       | ape fissioning Senat                                                                                                                                                                                                                                                                                                                                                                        |
| ![examples/00199971.jpg](./examples/00199971.jpg)                       | ling oughtlins near                                                                                                                                                                                                                                                                                                                                                                         |
| ![examples/multi-line_cn1.png](./examples/multi-line_cn1.png)           | 网络支付并无本质的区别，因为<br />每一个手机号码和邮件地址背后<br />都会对应着一个账户--这个账<br />户可以是信用卡账户、借记卡账<br />户，也包括邮局汇款、手机代<br />收、电话代收、预付费卡和点卡<br />等多种形式。                                                                                                                                                                                                                                                               |
| ![examples/multi-line_cn2.png](./examples/multi-line_cn2.png)           | 当然，在媒介越来越多的情形下,<br />意味着传播方式的变化。过去主流<br />的是大众传播,现在互动性和定制<br />性带来了新的挑战——如何让品牌<br />与消费者更加互动。                                                                                                                                                                                                                                                                                               |
| ![examples/multi-line_en_white.png](./examples/multi-line_en_white.png) | This chapter is currently only available <br />in this web version. ebook and print will follow.<br />Convolutional neural networks learn abstract <br />features and concepts from raw image pixels. Feature<br />Visualization visualizes the learned features <br />by activation maximization. Network Dissection labels<br />neural network units (e.g. channels) with human concepts. |
| ![examples/multi-line_en_black.png](./examples/multi-line_en_black.png) | transforms the image many times. First, the image <br />goes through many convolutional layers. In those<br />convolutional layers, the network learns new <br />and increasingly complex features in its layers. Then the <br />transformed image information goes through <br />the fully connected layers and turns into a classification<br />or prediction.                            |

## 详细使用说明

[类CnOcr](cnocr/cn_ocr.md) 是识别主类，包含了三个函数针对不同场景进行文字识别。类`CnOcr`的初始化函数如下：

```python
class CnOcr(object):
    def __init__(
        self,
        model_name: str = 'densenet_lite_136-fc',
        *,
        cand_alphabet: Optional[Union[Collection, str]] = None,
        context: str = 'cpu',  # ['cpu', 'gpu', 'cuda']
        model_fp: Optional[str] = None,
        model_backend: str = 'onnx',  # ['pytorch', 'onnx']
        root: Union[str, Path] = data_dir(),
        vocab_fp: Union[str, Path] = VOCAB_FP,
        **kwargs,
    )
```

其中的几个参数含义如下：

* `model_name`: 模型名称，即上面表格第一列中的值。默认为 `densenet_lite_136-fc`。更多可选模型见 [可直接使用的模型](models.md) 。

* `cand_alphabet`: 待识别字符所在的候选集合。默认为 `None`，表示不限定识别字符范围。取值可以是字符串，如 `"0123456789"`，或者字符列表，如 `["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]`。
  
  * `cand_alphabet`也可以初始化后通过类函数 `CnOcr.set_cand_alphabet(cand_alphabet)` 进行设置。这样同一个实例也可以指定不同的`cand_alphabet`进行识别。

* `context`：预测使用的机器资源，可取值为字符串`cpu`、`gpu`、`cuda:0`等。默认为 `cpu`。此参数仅在 `model_backend=='pytorch'` 时有效。

* `model_fp`:  如果不使用系统自带的模型，可以通过此参数直接指定所使用的模型文件（`.ckpt` 或 `.onnx` 文件）。

* `model_backend`：'pytorch', or 'onnx'。表明预测时是使用 `PyTorch` 版本模型，还是使用 `ONNX` 版本模型。 **同样的模型，ONNX 版本的预测速度一般是 PyTorch 版本的 2倍左右。** 默认为 'onnx'。

* `root`:  模型文件所在的根目录。
  
  * Linux/Mac下默认值为 `~/.cnocr`，表示模型文件所处文件夹类似 `~/.cnocr/2.1/densenet_lite_136-fc`。
  * Windows下默认值为 `C:\Users\<username>\AppData\Roaming\cnocr`。

* `vocab_fp`：字符集合的文件路径，即 `label_cn.txt` 文件路径。若训练的自有模型更改了字符集，看通过此参数传入新的字符集文件路径。

每个参数都有默认取值，所以可以不传入任何参数值进行初始化：`ocr = CnOcr()`。

---

类`CnOcr`主要包含三个函数，下面分别说明。

### 1. 函数`CnOcr.ocr(img_fp)`

函数`CnOcr.ocr(img_fp)`可以对包含多行文字（或单行）的图片进行文字识别。

**函数说明**：

- 输入参数 `img_fp`: 可以是需要识别的图片文件路径（如下例）；或者是已经从图片文件中读入的数组，类型可以为 `torch.Tensor` 或  `np.ndarray`，取值应该是`[0，255]`的整数，维数应该是 `[height, width]` （灰度图片）或者 `[height, width, channel]`，`channel` 可以等于`1`（灰度图片）或者`3`（`RGB`格式的彩色图片）。
- 返回值：为一个嵌套的`list`，其中的每个元素存储了对一行文字的识别结果，其中也包含了识别概率值。类似这样`[('第一行', 0.80), ('第二行', 0.75), ('第三行', 0.9)]`，其中的数字为对应的识别概率值。

**调用示例**：

```python
from cnocr import CnOcr

ocr = CnOcr()
res = ocr.ocr('examples/multi-line_cn1.png')
print("Predicted Chars:", res)
```

或：

```python
from cnocr.utils import read_img
from cnocr import CnOcr

ocr = CnOcr()
img_fp = 'examples/multi-line_cn1.png'
img = read_img(img_fp)
res = ocr.ocr(img)
print("Predicted Chars:", res)
```

上面使用的图片文件 [docs/examples/multi-line_cn1.png](./examples/multi-line_cn1.png)内容如下：

![examples/multi-line_cn1.png](./examples/multi-line_cn1.png)

上面预测代码段的返回结果如下：

```bash
Predicted Chars: [
    ('网络支付并无本质的区别，因为', 0.996096134185791), 
    ('每一个手机号码和邮件地址背后', 0.9903925061225891), 
    ('都会对应着一个账户一一这个账', 0.6401291489601135), 
    ('户可以是信用卡账户、借记卡账', 0.9446338415145874), 
    ('户，也包括邮局汇款、手机代', 0.9997618794441223), 
    ('收、电话代收、预付费卡和点卡', 0.7029080390930176), 
    ('等多种形式。', 0.8814011812210083)]
```

### 2. 函数`CnOcr.ocr_for_single_line(img_fp)`

如果明确知道要预测的图片中只包含了单行文字，可以使用函数`CnOcr.ocr_for_single_line(img_fp)`进行识别。和 `CnOcr.ocr()`相比，`CnOcr.ocr_for_single_line()`结果可靠性更强，因为它不需要做额外的分行处理。

**函数说明**：

- 输入参数 `img_fp`: 可以是需要识别的图片文件路径（如下例）；或者是已经从图片文件中读入的数组，类型可以为 `torch.Tensor` 或  `np.ndarray`，取值应该是`[0，255]`的整数，维数应该是 `[height, width]` （灰度图片）或者 `[height, width, channel]`，`channel` 可以等于`1`（灰度图片）或者`3`（`RGB`格式的彩色图片）。
- 返回值：为一个`tuple`，其中存储了对一行文字的识别结果，也包含了识别概率值。类似这样`('第一行', 0.80)`，其中的数字为对应的识别概率值。

**调用示例**：

```python
from cnocr import CnOcr

ocr = CnOcr()
res = ocr.ocr_for_single_line('examples/rand_cn1.png')
print("Predicted Chars:", res)
```

或：

```python
from cnocr.utils import read_img
from cnocr import CnOcr

ocr = CnOcr()
img_fp = 'examples/rand_cn1.png'
img = read_img(img_fp)
res = ocr.ocr_for_single_line(img)
print("Predicted Chars:", res)
```

对图片文件 [docs/examples/rand_cn1.png](./examples/rand_cn1.png)：

![examples/rand_cn1.png](./examples/rand_cn1.png)

的预测结果如下：

```bash
Predicted Chars: ('笠淡嘿骅谧鼎皋姚歼蠢驼耳窝挝涯狗蒽子犷', 0.34973764419555664)
```

### 3. 函数`CnOcr.ocr_for_single_lines(img_list, batch_size=1)`

函数`CnOcr.ocr_for_single_lines(img_list)`可以**对多个单行文字图片进行批量预测**。函数`CnOcr.ocr(img_fp)`和`CnOcr.ocr_for_single_line(img_fp)`内部其实都是调用的函数`CnOcr.ocr_for_single_lines(img_list)`。

**函数说明**：

- 输入参数` img_list`: 为一个`list`；其中每个元素可以是需要识别的图片文件路径（如下例）；或者是已经从图片文件中读入的数组，类型可以为 `torch.Tensor` 或  `np.ndarray`，取值应该是`[0，255]`的整数，维数应该是 `[height, width]` （灰度图片）或者 `[height, width, channel]`，`channel` 可以等于`1`（灰度图片）或者`3`（`RGB`格式的彩色图片）。
- 输入参数 `batch_size`: 待处理图片很多时，需要分批处理，每批图片的数量由此参数指定。默认为 `1`。
- 返回值：为一个嵌套的`list`，其中的每个元素存储了对一行文字的识别结果，其中也包含了识别概率值。类似这样`[('第一行', 0.80), ('第二行', 0.75), ('第三行', 0.9)]`，其中的数字为对应的识别概率值。

**调用示例**：

```python
import numpy as np

from cnocr.utils import read_img
from cnocr import CnOcr, line_split

ocr = CnOcr()
img_fp = 'examples/multi-line_cn1.png'
img = read_img(img_fp)
line_imgs = line_split(np.squeeze(img, -1), blank=True)
line_img_list = [line_img for line_img, _ in line_imgs]
res = ocr.ocr_for_single_lines(line_img_list)
print("Predicted Chars:", res)
```

更详细的使用方法，可参考 [tests/test_cnocr.py](https://github.com/breezedeus/cnocr/blob/master/tests/test_cnocr.py) 中提供的测试用例。

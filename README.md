English [README](./README_en.md) (`out-dated`).

# cnocr

**cnocr** 是 **Python 3** 下的**文字识别**（**Optical Character Recognition**，简称**OCR**）工具包，支持**中文**、**英文**的常见字符识别，自带了多个训练好的检测模型，安装后即可直接使用。欢迎扫码加入QQ交流群：

![QQ群二维码](./docs/cnocr-qq.jpg)



# 最近更新 【2021.08.26】：V2.0.0

主要变更：

* MXNet 越来越小众化，故从基于 MXNet 的实现转为基于 **PyTorch** 的实现；
* 重新实现了识别模型，优化了训练数据，重新训练模型；
* 优化了能识别的字符集合；
* 优化了对英文的识别效果；
* 优化了对场景文字的识别效果；
* 使用接口略有调整。



更多更新说明见 [RELEASE Notes](./RELEASE.md)。



## 使用场景说明

**cnocr** 主要针对的是**排版简单的印刷体文字图片**，如截图图片，扫描件等。目前内置的文字检测和分行模块无法处理复杂的文字排版定位。如果要用于场景文字图片的识别，需要结合其他的场景文字检测引擎使用，例如文字检测引擎 **[cnstd](https://github.com/breezedeus/cnstd)** 。



## 示例

| 图片                                                         | OCR结果                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![examples/helloworld.jpg](./examples/helloworld.jpg)        | Hello world!你好世界                                         |
| ![examples/chn-00199989.jpg](./examples/chn-00199989.jpg)    | 铑泡胭释邑疫反隽寥缔                                         |
| ![examples/chn-00199980.jpg](./examples/chn-00199980.jpg)    | 拇箬遭才柄腾戮胖惬炫                                         |
| ![examples/chn-00199984.jpg](./examples/chn-00199984.jpg)    | 寿猿嗅髓孢刀谎弓供捣                                         |
| ![examples/chn-00199985.jpg](./examples/chn-00199985.jpg)    | 马靼蘑熨距额猬要藕萼                                         |
| ![examples/chn-00199981.jpg](./examples/chn-00199981.jpg)    | 掉江悟厉励.谌查门蠕坑                                        |
| ![examples/00199975.jpg](./examples/00199975.jpg)            | nd-chips fructed ast                                         |
| ![examples/00199978.jpg](./examples/00199978.jpg)            | zouna unpayably Raqu                                         |
| ![examples/00199979.jpg](./examples/00199979.jpg)            | ape fissioning Senat                                         |
| ![examples/00199971.jpg](./examples/00199971.jpg)            | ling oughtlins near                                          |
| ![examples/multi-line_cn1.png](./examples/multi-line_cn1.png) | 网络支付并无本质的区别，因为<br />每一个手机号码和邮件地址背后<br />都会对应着一个账户--这个账<br />户可以是信用卡账户、借记卡账<br />户，也包括邮局汇款、手机代<br />收、电话代收、预付费卡和点卡<br />等多种形式。 |
| ![examples/multi-line_cn2.png](./examples/multi-line_cn2.png) | 当然，在媒介越来越多的情形下,<br />意味着传播方式的变化。过去主流<br />的是大众传播,现在互动性和定制<br />性带来了新的挑战——如何让品牌<br />与消费者更加互动。 |
| ![examples/multi-line_en_white.png](./examples/multi-line_en_white.png) | This chapter is currently only available in this web version. ebook and print will follow.<br />Convolutional neural networks learn abstract features and concepts from raw image pixels. Feature<br />Visualization visualizes the learned features by activation maximization. Network Dissection labels<br />neural network units (e.g. channels) with human concepts. |
| ![examples/multi-line_en_black.png](./examples/multi-line_en_black.png) | transforms the image many times. First, the image goes through many convolutional layers. In those<br />convolutional layers, the network learns new and increasingly complex features in its layers. Then the <br />transformed image information goes through the fully connected layers and turns into a classification<br />or prediction. |



## 安装

嗯，安装真的很简单。

```bash
pip install cnocr
```

> 注意：请使用 **Python3**（3.6以及之后版本应该都行），没测过Python2下是否ok。



## 可直接使用的模型

cnocr的ocr模型可以分为两阶段：第一阶段是获得ocr图片的局部编码向量，第二部分是对局部编码向量进行序列学习，获得序列编码向量。目前的PyTorch版本的两个阶段分别包含以下模型：

1. 局部编码模型（emb model）
   * **`densenet-s`**：一个小型的`densenet`网络；
2. 序列编码模型（seq model）
   * **`lstm`**：一层的LSTM网络；
   * **`gru`**：一层的GRU网络；
   * **`fc`**：两层的全连接网络。



cnocr **V2.0** 目前包含以下可直接使用的模型，训练好的模型都放在 **[cnocr-models](https://github.com/breezedeus/cnocr-models)** 项目中，可免费下载使用：

| 模型名称 | 局部编码模型 | 序列编码模型 | 模型大小 | 迭代次数 | 测试集准确率  |
| :------- | ------------ | ------------ | -------- | ------ | -------- |
| densenet-s-gru | densenet-s | gru | 11 M | 11 | 95.5% |
| densenet-s-fc | densenet-s | fc | 8.7 M | 39 | 91.9% |

> 模型名称是由局部编码模型和序列编码模型名称拼接而成。





## 使用方法

首次使用cnocr时，系统会**自动下载** zip格式的模型压缩文件，并存于 `~/.cnocr`目录（Windows下默认路径为 `C:\Users\<username>\AppData\Roaming\cnocr`）。
下载后的zip文件代码会自动对其解压，然后把解压后的模型相关目录放于`~/.cnocr/2.0`目录中。

如果系统无法自动成功下载zip文件，则需要手动从 **[cnocr-models](https://github.com/breezedeus/cnocr-models)** 下载此zip文件并把它放于 `~/.cnocr/2.0`目录。如果Github下载太慢，也可以从 [百度云盘](https://pan.baidu.com/s/1c68zjHfTVeqiSMXBEPYMrg) 下载， 提取码为 ` 9768`。

放置好zip文件后，后面的事代码就会自动执行了。



### 图片预测

类`CnOcr`是OCR的主类，包含了三个函数针对不同场景进行文字识别。类`CnOcr`的初始化函数如下：

```python
class CnOcr(object):
    def __init__(
        self,
        model_name: str = 'densenet-s-fc'
        model_epoch: Optional[int] = None,
        *,
        cand_alphabet: Optional[Union[Collection, str]] = None,
        context: str = 'cpu',  # ['cpu', 'gpu', 'cuda']
        model_fp: Optional[str] = None,
        root: Union[str, Path] = data_dir(),
        **kwargs,
    ):
```

其中的几个参数含义如下：

* `model_name`: 模型名称，即上面表格第一列中的值。默认为 `densenet-s-fc`。

* `model_epoch`: 模型迭代次数。默认为 `None`，表示使用默认的迭代次数值。对于模型名称 `densenet-s-fc`就是 `39`。

* `cand_alphabet`: 待识别字符所在的候选集合。默认为 `None`，表示不限定识别字符范围。取值可以是字符串，如 `"0123456789"`，或者字符列表，如 `["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]`。
  
   * `cand_alphabet`也可以初始化后通过类函数 `CnOcr.set_cand_alphabet(cand_alphabet)` 进行设置。这样同一个实例也可以指定不同的`cand_alphabet`进行识别。
   
* `context`：预测使用的机器资源，可取值为字符串`cpu`、`gpu`、`cuda:0`等。

* `model_fp`:  如果不使用系统自带的模型，可以通过此参数直接指定所使用的模型文件（`.ckpt` 文件）。

* `root`:  模型文件所在的根目录。

   * Linux/Mac下默认值为 `~/.cnocr`，表示模型文件所处文件夹类似 `~/.cnocr/2.0/densenet-s-fc`。
   * Windows下默认值为 `C:\Users\<username>\AppData\Roaming\cnocr`。

   

每个参数都有默认取值，所以可以不传入任何参数值进行初始化：`ocr = CnOcr()`。




类`CnOcr`主要包含三个函数，下面分别说明。



#### 1. 函数`CnOcr.ocr(img_fp)`

函数`CnOcr.ocr(img_fp)`可以对包含多行文字（或单行）的图片进行文字识别。



**函数说明**：

- 输入参数 `img_fp`: 可以是需要识别的图片文件路径（如下例）；或者是已经从图片文件中读入的数组，类型可以为 `torch.Tensor` 或  `np.ndarray`，取值应该是`[0，255]`的整数，维数应该是 `[height, width]` （灰度图片）或者 `[height, width, channel]`，`channel` 可以等于`1`（灰度图片）或者`3`（`RGB`格式的彩色图片）。
- 返回值：为一个嵌套的`list`，其中的每个元素存储了对一行文字的识别结果，其中也包含了识别概率值。类似这样`[(['第', '一', '行'], 0.80), (['第', '二', '行'], 0.75), (['第', '三', '行'], 0.9)]`，其中的数字为对应的识别概率值。



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



上面使用的图片文件 [examples/multi-line_cn1.png](./examples/multi-line_cn1.png)内容如下：

![examples/multi-line_cn1.png](./examples/multi-line_cn1.png)



上面预测代码段的返回结果如下：

```bash
Predicted Chars: [
		(['网', '络', '支', '付', '并', '无', '本', '质', '的', '区', '别', '，', '因', '为'], 0.8677546381950378), 
		(['每', '一', '个', '手', '机', '号', '码', '和', '邮', '件', '地', '址', '背', '后'], 0.6706454157829285), 
		(['都', '会', '对', '应', '着', '一', '个', '账', '户', '一', '一', '这', '个', '账'], 0.5052655935287476), 
		(['户', '可', '以', '是', '信', '用', '卡', '账', '户', '、', '借', '记', '卡', '账'], 0.7785991430282593), 
		(['户', '，', '也', '包', '括', '邮', '局', '汇', '款', '、', '手', '机', '代'], 0.37458470463752747), 
		(['收', '、', '电', '话', '代', '收', '、', '预', '付', '费', '卡', '和', '点', '卡'], 0.7326119542121887), 
		(['等', '多', '种', '形', '式', '。'], 0.14462216198444366)]
```



#### 2. 函数`CnOcr.ocr_for_single_line(img_fp)`

如果明确知道要预测的图片中只包含了单行文字，可以使用函数`CnOcr.ocr_for_single_line(img_fp)`进行识别。和 `CnOcr.ocr()`相比，`CnOcr.ocr_for_single_line()`结果可靠性更强，因为它不需要做额外的分行处理。

**函数说明**：

- 输入参数 `img_fp`: 可以是需要识别的图片文件路径（如下例）；或者是已经从图片文件中读入的数组，类型可以为 `torch.Tensor` 或  `np.ndarray`，取值应该是`[0，255]`的整数，维数应该是 `[height, width]` （灰度图片）或者 `[height, width, channel]`，`channel` 可以等于`1`（灰度图片）或者`3`（`RGB`格式的彩色图片）。
- 返回值：为一个`tuple`，其中存储了对一行文字的识别结果，也包含了识别概率值。类似这样`(['第', '一', '行'], 0.80)`，其中的数字为对应的识别概率值。



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


对图片文件 [examples/rand_cn1.png](./examples/rand_cn1.png)：

![examples/rand_cn1.png](./examples/rand_cn1.png)

的预测结果如下：

```bash
Predicted Chars: (['笠', '淡', '嘿', '骅', '谧', '鼎', '皋', '姚', '歼', '蠢', '驼', '耳', '胬', '挝', '涯', '狗', '蒽', '了', '狞'], 0.7832438349723816)
```



#### 3. 函数`CnOcr.ocr_for_single_lines(img_list)`

函数`CnOcr.ocr_for_single_lines(img_list)`可以**对多个单行文字图片进行批量预测**。函数`CnOcr.ocr(img_fp)`和`CnOcr.ocr_for_single_line(img_fp)`内部其实都是调用的函数`CnOcr.ocr_for_single_lines(img_list)`。



**函数说明**：

- 输入参数` img_list`: 为一个`list`；其中每个元素可以是需要识别的图片文件路径（如下例）；或者是已经从图片文件中读入的数组，类型可以为 `torch.Tensor` 或  `np.ndarray`，取值应该是`[0，255]`的整数，维数应该是 `[height, width]` （灰度图片）或者 `[height, width, channel]`，`channel` 可以等于`1`（灰度图片）或者`3`（`RGB`格式的彩色图片）。
- 返回值：为一个嵌套的`list`，其中的每个元素存储了对一行文字的识别结果，其中也包含了识别概率值。类似这样`[(['第', '一', '行'], 0.80), (['第', '二', '行'], 0.75), (['第', '三', '行'], 0.9)]`，其中的数字为对应的识别概率值。



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



更详细的使用方法，可参考 [tests/test_cnocr.py](./tests/test_cnocr.py) 中提供的测试用例。



### 结合文字检测引擎 **[cnstd](https://github.com/breezedeus/cnstd)** 使用

对于一般的场景图片（如照片、票据等），需要先利用场景文字检测引擎 **[cnstd](https://github.com/breezedeus/cnstd)** 定位到文字所在位置，然后再利用 **cnocr** 进行文本识别。

```python
from cnstd import CnStd
from cnocr import CnOcr

std = CnStd()
cn_ocr = CnOcr()

box_infos = std.detect('examples/taobao.jpg')

for box_info in box_infos['detected_texts']:
    cropped_img = box_info['cropped_img']
    ocr_res = cn_ocr.ocr_for_single_line(cropped_img)
    print('ocr result: %s' % str(ocr_out))
```

注：运行上面示例需要先安装 **[cnstd](https://github.com/breezedeus/cnstd)** ：

```bash
pip install cnstd
```

**[cnstd](https://github.com/breezedeus/cnstd)** 相关的更多使用说明请参考其项目地址。





### 脚本使用

**cnocr** 包含了几个命令行工具，安装 **cnocr** 后即可使用。



#### 预测单个文件或文件夹中所有图片

使用命令 **`cnocr predict`** 预测单个文件或文件夹中所有图片，以下是使用说明：



```bash
(venv) ➜  cnocr git:(pytorch) ✗ cnocr predict -h
Usage: cnocr predict [OPTIONS]

Options:
  -m, --model-name [densenet-s-lstm|densenet-s-gru|densenet-s-fc]
                                  模型名称。默认值为 densenet-s-fc
  --model_epoch INTEGER           model epoch。默认为 `None`，表示使用系统自带的预训练模型
  -p, --pretrained-model-fp TEXT  使用训练好的模型。默认为 `None`，表示使用系统自带的预训练模型
  --context TEXT                  使用cpu还是 `gpu` 运行代码，也可指定为特定gpu，如`cuda:0`。默认为
                                  `cpu`

  -i, --img-file-or-dir TEXT      输入图片的文件路径或者指定的文件夹  [required]
  -s, --single-line               是否输入图片只包含单行文字。对包含单行文字的图片，不做按行切分；否则会先对图片按行分割后
                                  再进行识别

  -h, --help                      Show this message and exit.
```



例如可以使用以下命令对图片 `examples/rand_cn1.png` 进行文字识别：

```bash
cnstd predict -i examples/rand_cn1.png -s
```



具体使用也可参考文件 [Makefile](./Makefile) 。



#### 模型训练

使用命令 **`cnocr train`**  训练文本检测模型，以下是使用说明：

```bash
(venv) ➜  cnocr git:(pytorch) ✗ cnocr train -h
Usage: cnocr train [OPTIONS]

Options:
  -m, --model-name [densenet-s-fc|densenet-s-lstm|densenet-s-gru]
                                  模型名称。默认值为 densenet-s-fc
  -i, --index-dir TEXT            索引文件所在的文件夹，会读取文件夹中的 train.tsv 和 dev.tsv 文件
                                  [required]

  --train-config-fp TEXT          训练使用的json配置文件，参考 `example/train_config.json`
                                  [required]

  -r, --resume-from-checkpoint TEXT
                                  恢复此前中断的训练状态，继续训练。默认为 `None`
  -p, --pretrained-model-fp TEXT  导入的训练好的模型，作为初始模型。优先级低于"--restore-training-
                                  fp"，当传入"--restore-training-fp"时，此传入失效。默认为
                                  `None`

  -h, --help                      Show this message and exit.
```



例如可以使用以下命令进行训练：

```bash
cnocr train -m densenet-s-fc --index-dir data/test --train-config-fp examples/train_config.json
```



训练数据的格式见文件夹 [data/test](./data/test) 中的  [train.tsv](./data/test/train.tsv) 和 [dev.tsv](./data/test/dev.tsv) 文件。



具体使用也可参考文件 [Makefile](./Makefile) 。



#### 模型转存

训练好的模型会存储训练状态，使用命令 **`cnocr resave`**  去掉与预测无关的数据，降低模型大小。

```bash
(venv) ➜  cnocr git:(pytorch) ✗ cnocr resave -h
Usage: cnocr resave [OPTIONS]

  训练好的模型会存储训练状态，使用此命令去掉预测时无关的数据，降低模型大小

Options:
  -i, --input-model-fp TEXT   输入的模型文件路径  [required]
  -o, --output-model-fp TEXT  输出的模型文件路径  [required]
  -h, --help                  Show this message and exit.
```







## 未来工作

* [x] 支持图片包含多行文字 (`Done`)
* [x] crnn模型支持可变长预测，提升灵活性 (since `V1.0.0`)
* [x] 完善测试用例 (`Doing`)
* [x] 修bugs（目前代码还比较凌乱。。） (`Doing`)
* [x] 支持`空格`识别（since `V1.1.0`）
* [x] 尝试新模型，如 DenseNet，进一步提升识别准确率（since `V1.1.0`）
* [x] 优化训练集，去掉不合理的样本；在此基础上，重新训练各个模型
* [x] 由 MXNet 改为 PyTorch 架构（since `V2.0.0`）
* [ ] 基于 PyTorch 训练更高效的模型
* [ ] 支持列格式的文字识别


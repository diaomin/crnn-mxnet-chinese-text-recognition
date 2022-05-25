# CnOcr

[**CnOcr**](https://github.com/breezedeus/cnocr) 是 **Python 3** 下的**文字识别**（**Optical Character Recognition**，简称**OCR**）工具包，支持**简体中文**、**繁体中文**（部分模型）、**英文**和**数字**的常见字符识别，支持竖排文字的识别。自带了**10+个** [训练好的识别模型](models.md)，安装后即可直接使用。欢迎扫码加入[QQ交流群](contact.md)。

作者也维护 **知识星球** [**CnOCR/CnSTD私享群**](https://t.zsxq.com/FEYZRJQ) ，欢迎加入。**知识星球私享群**会陆续发布一些CnOCR/CnSTD相关的私有资料，包括**更详细的训练教程**，**未公开的模型**，使用过程中遇到的难题解答等。本群也会发布OCR/STD相关的最新研究资料。此外，**私享群中作者每月提供两次免费特有数据的训练服务**。

CnOcr的目标是**使用简单**。

可以使用 [**在线 Demo**](demo.md) 查看效果。

## 安装简单

嗯，安装真的很简单。

```bash
pip install cnocr
```

更多说明可见 [安装文档](install.md)。

## 使用简单

使用 `CnOcr.ocr()` 识别下图：

![多行文字图片](examples/multi-line_cn1.png)

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

返回结果如下：

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

更多说明可见 [使用方法](usage.md)。

## 命令行工具

具体见 [命令行工具](command.md)。

### 训练自己的模型

具体见 [模型训练](train.md)。

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

## 其他文档

* [场景文字识别技术介绍（PPT+视频）](std_ocr.md)
* 对于通用场景的文字识别，使用 [文本检测CnStd + 文字识别CnOcr](cnstd_cnocr.md)
* [RELEASE文档](RELEASE.md)

## 未来工作

* [x] 支持图片包含多行文字 (`Done`)
* [x] crnn模型支持可变长预测，提升灵活性 (since `V1.0.0`)
* [x] 完善测试用例 (`Doing`)
* [x] 修bugs（目前代码还比较凌乱。。） (`Doing`)
* [x] 支持`空格`识别（since `V1.1.0`）
* [x] 尝试新模型，如 DenseNet，进一步提升识别准确率（since `V1.1.0`）
* [x] 优化训练集，去掉不合理的样本；在此基础上，重新训练各个模型
* [x] 由 MXNet 改为 PyTorch 架构（since `V2.0.0`）
* [x] 基于 PyTorch 训练更高效的模型
* [x] 支持列格式的文字识别
- [ ] 支持与 [CnStd](https://github.com/breezedeus/cnstd) 的无缝衔接



---

官方代码库：[https://github.com/breezedeus/cnocr](https://github.com/breezedeus/cnocr)。

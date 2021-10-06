# CnOcr

**[CnOcr](https://github.com/breezedeus/cnocr)** 是 **Python 3** 下的**文字识别**（**Optical Character Recognition**，简称**OCR**）工具包，
支持**中文**、**英文**的常见字符识别，自带了多个[训练好的识别模型](models.md)，安装后即可直接使用。
欢迎扫码加入[QQ交流群](contact.md)。

CnOcr的目标是**使用简单**。

可以使用 [在线 Demo](demo.md) 查看效果。


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

更多说明可见 [使用方法](usage.md)。


## 命令行工具

具体见 [命令行工具](command.md)。

### 训练自己的模型

具体见 [模型训练](train.md)。

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
* [ ] 基于 PyTorch 训练更高效的模型
* [ ] 支持列格式的文字识别


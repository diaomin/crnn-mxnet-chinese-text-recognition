# CnOcr

**CnOcr** 是 **Python 3** 下的**文字识别**（**Optical Character Recognition**，简称**OCR**）工具包，
支持**中文**、**英文**的常见字符识别，自带了多个[训练好的识别模型](models.md)，安装后即可直接使用。
欢迎扫码加入[QQ交流群](contact.md)。

CnOcr的目标是**用起来简单**。

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


## 其他文档

* [场景文字识别介绍](std_ocr.md)
* [文本检测CnStd + 文字识别CnOcr](cnstd_cnocr.md)
* [RELEASE文档](RELEASE.md)


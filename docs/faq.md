# 常见问题（FAQ）

## CnOcr 是免费的吗？

CnOcr是免费的，而且是开源的。可以按需自行调整发布或商业使用。

## CnOcr 能识别英文以及空格吗？

可以。

## CnOcr 能识别繁体中文吗？

部分模型可以，具体见 [可用模型](models.md)。

## CnOcr 能识别竖排文字的图片吗？

部分模型可以，具体见 [可用模型](models.md)。

## CnOcr能支持其他语言的模型吗？

暂时没有。如有其他外语（如日、韩等）识别需求，可在 **知识星球** [**CnOCR/CnSTD私享群**](https://t.zsxq.com/FEYZRJQ) 中向作者提出建议。



## 文本检测的部分结果翻转了180度

CnOCR 中已支持**角度判断功能**，可通过开启此功能来修正检测文本翻转180度的问题。`CnOcr` 初始化时传入以下参数即可开启角度判断功能。

```python
from cnocr import CnOcr

img_fp = './docs/examples/huochepiao.jpeg'
ocr = CnOcr(det_more_configs={'use_angle_clf': True})  # 开启角度判断功能
out = ocr.ocr(img_fp)

print(out)
```

具体可参考 [CnSTD 文档](https://github.com/breezedeus/cnstd) 。

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

## CnStd检测的部分结果可能翻转180度

1. 如果明确知道检测图片中不包含倾斜的文字，建议在初始化 `CnStd`时传入参数 `rotated_bbox=False`，详见文档 [GitHub - breezedeus/cnstd](https://github.com/breezedeus/cnstd) 。

2. **OpenCV的版本号也会会带来翻转问题**，cnstd官方使用的版本是 `4.2.0.34`，推荐安装此版本或与此临近的版本。
   
   ```bash
   > pip show opencv-python
   
   Name: opencv-python
   Version: 4.2.0.34
   ```

    如果安装了 **`4.5.2`** 及之后的`opencv-python`版本，在 `rotated_bbox=True` 时可能会报错，此问题后续版本会修复。

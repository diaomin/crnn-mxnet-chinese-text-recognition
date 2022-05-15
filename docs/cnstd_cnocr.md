# 强强联合：[CnStd](https://github.com/breezedeus/cnstd) + CnOcr

关于为什么要结合 [CnStd](https://github.com/breezedeus/cnstd) 和 CnOcr 一起使用，可参考 [场景文字识别介绍](std_ocr.md)。

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
    print('ocr result: %s' % str(ocr_res))
```

注：运行上面示例需要先安装 **[cnstd](https://github.com/breezedeus/cnstd)** ：

```bash
pip install cnstd
```

**[cnstd](https://github.com/breezedeus/cnstd)** 相关的更多使用说明请参考其项目地址。

可基于 [在线 Demo](demo.md) 查看 CnStd + CnOcr 的联合效果。

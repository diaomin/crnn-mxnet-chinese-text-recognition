## 安装

嗯，顺利的话一行命令即可。

```bash
pip install cnocr
```

安装速度慢的话，可以指定国内的安装源，如使用豆瓣源：

```bash
pip install cnocr -i https://pypi.doubanio.com/simple
```

> **Note**
>
> 请使用 **Python3**（3.6以及之后版本应该都行），没测过Python2下是否ok。



> **Warning** 
>
> 如果电脑中从未安装过 `PyTorch`，`OpenCV` python包，初次安装可能会遇到问题，但一般都是常见问题，可以自行百度/Google解决。



如遇到安装问题，欢迎在 [Github](https://github.com/breezedeus/cnocr)、 知识星球[**CnOCR/CnSTD私享群**](https://t.zsxq.com/FEYZRJQ) 或者 微信交流群反馈给作者 [breezedeus](https://github.com/breezedeus) 。





### GPU 环境使用 ONNX 模型

默认情况下安装的 **ONNX** 包是 **`onnxruntime`**，它只能在 `CPU` 上运行。如果需要在 `GPU` 环境使用 **ONNX** 模型，需要卸载此包，然后安装包 **`onnxruntime-gpu`** 。

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

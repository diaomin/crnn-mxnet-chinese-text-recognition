中文版说明请见[中文README](./README_cn.md)。

# cnocr
A python package for Chinese OCR with available trained models.
So it can be used directly after installed.

The accuracy of the current crnn model is about `98.7%`.

The project originates from our own ([爱因互动 Ein+](https://einplus.cn)) internal needs.
Thanks for the internal supports.

## Changes

Most of the codes are adapted from [crnn-mxnet-chinese-text-recognition](https://github.com/diaomin/crnn-mxnet-chinese-text-recognition).
Much thanks to the author.

Some changes are:

* use raw MXNet CTC Loss instead of WarpCTC Loss. No more complicated installation.
* public pre-trained model for anyone. No more a-few-days training.
* add online `predict` function and script. Easy to use.

## Installation

```bash
pip install cnocr
```

> Please use Python3 (3.4, 3.5, 3.6 should work). Python2 is not tested.

## Usage

### Predict

```python
from cnocr import CnOcr
ocr = CnOcr()
res = ocr.ocr('examples/multi-line_cn1.png')
print("Predicted Chars:", res)
```

When you run the previous codes, the model files will be downloaded automatically from 
[Dropbox](https://www.dropbox.com/s/5n09nxf4x95jprk/cnocr-models-v0.1.0.zip) to `~/.cnocr`. 
The zip file will be extracted and you can find the resulting model files in `~/.cnocr/models` by default.
In case the automatic download can't perform well, you can download the zip file manually 
from [Baidu NetDisk](https://pan.baidu.com/s/1s91985r0YBGbk_1cqgHa1Q) with extraction code `pg26`,
and put the zip file to `~/.cnocr`. The code will do else.

Try the predict command for [examples/multi-line_cn1.png](./examples/multi-line_cn1.png):

![examples/multi-line_cn1.png](./examples/multi-line_cn1.png)

```bash
python scripts/cnocr_predict.py --file examples/multi-line_cn1.png
```
You will get:
```python
Predicted Chars: [['网', '络', '支', '付', '并', '无', '本', '质', '的', '区', '别', '，', '因', '为'], ['每', '一', '个', '手', '机', '号', '码', '和', '邮', '件', '地', '址', '背', '后'], ['都', '会', '对', '应', '着', '一', '个', '账', '户', '一', '一', '这', '个', '账'], ['户', '可', '以', '是', '信', '用', '卡', '账', '户', '、', '借', '记', '卡', '账'], ['户', '，', '也', '包', '括', '邮', '局', '汇', '款', '、', '手', '机', '代'], ['收', '、', '电', '话', '代', '收', '、', '预', '付', '费', '卡', '和', '点', '卡'], ['等', '多', '种', '形', '式', '。']]
```



### Predict for Single-line-characters Image

If you know your image includes only one single line characters, you can use function `Cnocr.ocr_for_single_line()` instead of  `Cnocr.ocr()`.  `Cnocr.ocr_for_single_line()` should be more efficient.

```python
from cnocr import CnOcr
ocr = CnOcr()
res = ocr.ocr_for_single_line('examples/rand_cn1.png')
print("Predicted Chars:", res)
```

With file [examples/multi-line_cn1.png](./examples/multi-line_cn1.png)：

![examples/rand_cn1.png](./examples/rand_cn1.png)

you will get:

```python
Predicted Chars: ['笠', '淡', '嘿', '骅', '谧', '鼎', '皋', '姚', '歼', '蠢', '驼', '耳', '胬', '挝', '涯', '狗', '蒽', '子', '犷']
```



### (No NECESSARY) Train

You can use the package without any train. But if you really really want to train your own models,
follow this:

```bash
python scripts/cnocr_train.py --cpu 2 --num_proc 4 --loss ctc --dataset cn_ocr
```

## Future Work
* [x] support multi-line-characters recognition
* Support space recognition
* Bugfixes
* Add Tests
* Maybe use no symbol to rewrite the model
* Try other models such as DenseNet, ResNet

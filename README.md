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
res = ocr.ocr_for_single_line('examples/rand_cn1.png')
print("Predicted Chars:", res)
```

When you run the previous codes, the model files will be downloaded automatically from 
[Dropbox](https://www.dropbox.com/s/5n09nxf4x95jprk/cnocr-models-v0.1.0.zip) to `~/.cnocr`. 
The zip file will be extracted and you can find the resulting model files in '~/.cnocr/models' by default.
In case the automatic download can't perform well, you can download the zip file manually 
from [Baidu NetDisk](https://pan.baidu.com/s/1s91985r0YBGbk_1cqgHa1Q) with extraction code `pg26`,
and put the zip file to `~/.cnocr`. The code will do else.

Try the predict command for [examples/rand_cn1.png](./examples/rand_cn1.png):

![examples/rand_cn1.png](./examples/rand_cn1.png)

```bash
python scripts/cnocr_predict.py --file examples/rand_cn1.png
```
You will get:
```bash
Predicted Chars: ['笠', '淡', '嘿', '骅', '谧', '鼎', '皋', '姚', '歼', '蠢', '驼', '耳', '胬', '挝', '涯', '狗', '蒽', '子', '犷']
```

### (No NECESSARY) Train

You can use the package without any train. But if you really really want to train your own models,
follow this:

```bash
python scripts/cnocr_train.py --cpu 2 --num_proc 4 --loss ctc --dataset cn_ocr
```

## Future Work
* Support space recognition
* Bugfixes
* Add Tests
* Maybe use no symbol to rewrite the model
* Try other models such as DenseNet, ResNet

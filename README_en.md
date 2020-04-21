中文版说明请见[中文README](./README.md)。



# Update 2019.07.25: release cnocr V1.0.0

`cnocr` `v1.0.0` is released, which is more efficient for prediction. **The new version of the model is not compatible with the previous version.** So if upgrading, please download the latest model file again. See below for the details (same as before).



Main changes are：

-  **The new crnn model supports prediction for variable-width image files, so is more efficient for prediction.**
-  Support fine-tuning the existing model with specific data.
-  Fix bugs，such as `train accuracy` always `0`.
-  Depended package `mxnet` is upgraded from `1.3.1`  to `1.4.1`.



# cnocr

A python package for Chinese OCR with available trained models.
So it can be used directly after installed.

The accuracy of the current crnn model is about `98.8%`.

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

The first time cnocr is used, the model files will be downloaded automatically from 
[Dropbox](https://www.dropbox.com/s/7w8l3mk4pvkt34w/cnocr-models-v1.0.0.zip?dl=0) to `~/.cnocr`. 

The zip file will be extracted and you can find the resulting model files in `~/.cnocr/models` by default.
In case the automatic download can't perform well, you can download the zip file manually 
from [Baidu NetDisk](https://pan.baidu.com/s/1DWV3H2UWmzOU6d48UbTYVw) with extraction code `ss81`, and put the zip file to `~/.cnocr`. The code will do else.



### Predict

Three functions are provided for prediction.



#### 1. `CnOcr.ocr(img_fp)`

The function `cnOcr.ocr (img_fp)` can recognize texts in an image containing multiple lines of text (or single lines).



**Function Description**

- input parameter `img_fp`: image file path; or color image `mx.nd.NDArray` or `np.ndarray`, with shape `(height, width, 3)`, and the channels should be RGB formatted.
- return: `List(List(Char))`,  such as:  `[['第', '一', '行'], ['第', '二', '行'], ['第', '三', '行']]`.
  



**Use Case**


```python
from cnocr import CnOcr
ocr = CnOcr()
res = ocr.ocr('examples/multi-line_cn1.png')
print("Predicted Chars:", res)
```

or:

```python
import mxnet as mx
from cnocr import CnOcr
ocr = CnOcr()
img_fp = 'examples/multi-line_cn1.png'
img = mx.image.imread(img_fp, 1)
res = ocr.ocr(img)
print("Predicted Chars:", res)
```

The previous codes can recognize texts in the image file [examples/multi-line_cn1.png](./examples/multi-line_cn1.png):

![examples/multi-line_cn1.png](./examples/multi-line_cn1.png)

The OCR results shoule be:

```bash
Predicted Chars: [['网', '络', '支', '付', '并', '无', '本', '质', '的', '区', '别', '，', '因', '为'],
                  ['每', '一', '个', '手', '机', '号', '码', '和', '邮', '件', '地', '址', '背', '后'],
                  ['都', '会', '对', '应', '着', '一', '个', '账', '户', '一', '―', '这', '个', '账'],
                  ['户', '可', '以', '是', '信', '用', '卡', '账', '户', '、', '借', '记', '卡', '账'],
                  ['户', '，', '也', '包', '括', '邮', '局', '汇', '款', '、', '手', '机', '代'],
                  ['收', '、', '电', '话', '代', '收', '、', '预', '付', '费', '卡', '和', '点', '卡'],
                  ['等', '多', '种', '形', '式', '。']]
```

#### 2. `CnOcr.ocr_for_single_line(img_fp)`

If you know that the image you're predicting contains only one line of text, function `CnOcr.ocr_for_single_line(img_fp)` can be used instead。Compared with `CnOcr.ocr()`, the result of `CnOcr.ocr_for_single_line()` is more reliable because the process of splitting lines is not required. 



**Function Description**

- input parameter `img_fp`: image file path; or color image `mx.nd.NDArray` or `np.ndarray`, with shape `[height, width]` or `[height, width, channel]`.  The optional channel should be `1` (gray image) or `3` (color image).
- return: `List(Char)`,  such as:  `['你', '好']`.



**Use Case**：

```python
from cnocr import CnOcr
ocr = CnOcr()
res = ocr.ocr_for_single_line('examples/rand_cn1.png')
print("Predicted Chars:", res)
```

or:

```python
import mxnet as mx
from cnocr import CnOcr
ocr = CnOcr()
img_fp = 'examples/rand_cn1.png'
img = mx.image.imread(img_fp, 1)
res = ocr.ocr_for_single_line(img)
print("Predicted Chars:", res)
```


The previous codes can recognize texts in the image file  [examples/rand_cn1.png](./examples/rand_cn1.png)：

![examples/rand_cn1.png](./examples/rand_cn1.png)

The OCR results shoule be:

```bash
Predicted Chars: ['笠', '淡', '嘿', '骅', '谧', '鼎', '臭', '姚', '歼', '蠢', '驼', '耳', '裔', '挝', '涯', '狗', '蒽', '子', '犷'] 
```

#### 3. `CnOcr.ocr_for_single_lines(img_list)`

Function `CnOcr.ocr_for_single_lines(img_list)` can predict a number of single-line-text image arrays batchly. Actually `CnOcr.ocr(img_fp)` and `CnOcr.ocr_for_single_line(img_fp)` both invoke `CnOcr.ocr_for_single_lines(img_list)` internally.



**Function Description**

- input parameter `img_list`: list of images, in which each element should be a line image array,  with type `mx.nd.NDArray` or `np.ndarray`.  Each element should be a tensor with values ranging from `0` to` 255`, and with shape `[height, width]` or `[height, width, channel]`.  The optional channel should be `1` (gray image) or `3` (color image).
- return: `List(List(Char))`,  such as:  `[['第', '一', '行'], ['第', '二', '行'], ['第', '三', '行']]`.



**Use Case**：

```python
import mxnet as mx
from cnocr import CnOcr
ocr = CnOcr()
img_fp = 'examples/multi-line_cn1.png'
img = mx.image.imread(img_fp, 1).asnumpy()
line_imgs = line_split(img, blank=True)
line_img_list = [line_img for line_img, _ in line_imgs]
res = ocr.ocr_for_single_lines(line_img_list)
print("Predicted Chars:", res)
```

More use cases can be found at [tests/test_cnocr.py](./tests/test_cnocr.py).


### Using  the Script

```bash
python scripts/cnocr_predict.py --file examples/multi-line_cn1.png
```



### (No NECESSARY) Train

You can use the package without any train. But if you really really want to train your own models, follow this:

```bash
python scripts/cnocr_train.py --cpu 2 --num_proc 4 --loss ctc --dataset cn_ocr
```



Fine-tuning the model with specific data from existing models is also supported. Please refer to the following command:

```bash
python scripts/cnocr_train.py --cpu 2 --num_proc 4 --loss ctc --dataset cn_ocr --load_epoch 20
```



More references can be found at  [scripts/run_cnocr_train.sh](./scripts/run_cnocr_train.sh).



## Future Work

* [x] support multi-line-characters recognition (`Done`)
* [x] crnn model supports prediction for variable-width image files (`Done`)
* [x] Add Unit Tests  (`Doing`)
* [x]  Bugfixes  (`Doing`)
* [ ] Support space recognition (Tried, but not successful for now )
* [ ] Try other models such as DenseNet, ResNet

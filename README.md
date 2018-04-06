# crnn-mxnet-chinese-text-recognition
This is an implementation of CRNN (CNN+LSTM+CTC) for chinese text recognition.

## Building MXNet with warp-ctc
1. In order to use `mxnet.symbol.WarpCTC` layer, you need to first build Baidu's [warp-ctc](https://github.com/baidu-research/warp-ctc) library from source 
2. Then build MXNet from source with warp-ctc config flags enabled.

## Data Preparation
1. Download the [Synthetic Chinese Dataset](https://pan.baidu.com/s/1dFda6R3)(contributed by https://github.com/senlinuc/caffe_ocr)
   
   A glance of the dataset 
   *`almost 3.6 million synthetic chinese text images.
   *`5,990 different categories in total.
   *`Each image has a length of 10 characters. 
   
2. Create train.txt and text.txt with the format like this:  
```
           image_name1 label1_1 label1_2 label1_3...
           image_name2 label2_1 label2_2 label2_3...
```
Optional: downoad the two files [here](https://pan.baidu.com/s/1xQ38TTUrxMytVp1VY6Y4Pg)

## Training
1. Modify the path of images and txt files in train.py 
2. Run
```
$ python train.py 2>&1 | tee log.txt
```
3. After almost 19 epoches, you can get 99.0502% validation accuracy.
```
2018-04-01 03:35:35,136 Epoch[18] Batch [25450]	Speed: 53.10 samples/sec	accuracy=0.988125
2018-04-01 03:37:37,482 Epoch[18] Batch [25500]	Speed: 52.31 samples/sec	accuracy=0.986719
2018-04-01 03:39:38,613 Epoch[18] Batch [25550]	Speed: 52.84 samples/sec	accuracy=0.989531
2018-04-01 03:41:40,470 Epoch[18] Batch [25600]	Speed: 52.52 samples/sec	accuracy=0.987969
2018-04-01 03:42:27,544 Epoch[18] Train-accuracy=0.988672
2018-04-01 03:42:27,544 Epoch[18] Time cost=80796.510
2018-04-01 03:42:27,610 Saved checkpoint to "./check_points/model-0019.params"
2018-04-01 05:34:43,096 Epoch[18] Validation-accuracy=0.990502
```
Hare is a [pre-trained model](https://pan.baidu.com/s/1iwOVZJxF-P14LemziisLwA) you can download directly.

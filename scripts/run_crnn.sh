#!/usr/bin/env bash
# -*- coding: utf-8 -*-

cd `dirname $0`

# 训练中文ocr模型crnn
python scripts/cnocr_train.py --cpu 2 --num_proc 4 --loss ctc --dataset cn_ocr


## 预测中文图片
#python scripts/cnocr_predict.py --file examples/rand_cn1.png
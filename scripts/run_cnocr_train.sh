#!/usr/bin/env bash
# -*- coding: utf-8 -*-

cd `dirname $0`/../

## 训练captcha
#python scripts/cnocr_train.py --cpu 2 --num_proc 2 --loss ctc --dataset captcha --font_path /Users/king/Documents/WhatIHaveDone/Test/text_renderer/data/fonts/chn/msyh.ttf

# 训练中文ocr模型crnn
python scripts/cnocr_train.py --cpu 2 --num_proc 4 --loss ctc --dataset cn_ocr

## gpu版本
#python scripts/cnocr_train.py --gpu 1 --num_proc 8 --loss ctc --dataset cn_ocr --data_root /jfs/jinlong/data/ocr/outer/images \
#    --train_file /jfs/jinlong/data/ocr/outer/train.txt --test_file /jfs/jinlong/data/ocr/outer/test.txt

## 预测中文图片
#python scripts/cnocr_predict.py --file examples/rand_cn1.png
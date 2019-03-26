#!/usr/bin/env bash
# -*- coding: utf-8 -*-

cd `dirname $0`

# 训练中文ocr模型crnn
python train_ocr.py --cpu 2 --num_proc 4 --loss ctc --dataset cn_ocr

gen-lst:
	python scripts/im2rec.py --list --chunks 1 --train-idx-fp data/sample-data/train.txt --test-idx-fp data/sample-data/test.txt --prefix data/lst/sample-data

gen-rec:
	python scripts/im2rec.py --pack-label --color 1 --num-thread 1 --prefix data/lst --root data/sample-data

train:
	python scripts/cnocr_train.py --cpu 2 --loss ctc --dataset cn_ocr --train_file data/lst/sample-data_train --test_file data/lst/sample-data_test

predict:
	python scripts/cnocr_predict.py --file examples/rand_cn1.png



.PHONY: gen-lst gen-rec train predict

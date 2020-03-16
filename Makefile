# ['conv-rnn', 'conv-rnn-lite', 'densenet-rnn', 'densenet-rnn-lite']
MODEL_NAME = densenet-rnn-lite

gen-lst:
	python scripts/im2rec.py --list --num-label 20 --chunks 1 --train-idx-fp data/sample-data/train.txt --test-idx-fp data/sample-data/test.txt --prefix data/lst/sample-data

gen-rec:
	python scripts/im2rec.py --pack-label --color 1 --num-thread 1 --prefix data/lst --root data/sample-data

train:
	python scripts/cnocr_train.py --cpu 2 --loss ctc --model_name $(MODEL_NAME) --train_file data/lst/sample-data_train --test_file data/lst/sample-data_test

evaluate:
	python scripts/cnocr_evaluate.py -v -i data/sample-data/test.txt --image-prefix-dir data/sample-data --batch-size 128 -o evaluate.out

predict:
	python scripts/cnocr_predict.py --file examples/rand_cn1.png


.PHONY: gen-lst gen-rec train evaluate predict

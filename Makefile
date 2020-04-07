# ['conv', 'conv-lite-rnn', 'densenet', 'densenet-lite']
EMB_MODEL_TYPE = densenet-lite
SEQ_MODEL_TYPE = fc
MODEL_NAME = $(EMB_MODEL_TYPE)-$(SEQ_MODEL_TYPE)

gen-lst:
	python scripts/im2rec.py --list --num-label 20 --chunks 1 --train-idx-fp data/selected/train.txt --test-idx-fp data/selected/test.txt --prefix data/selected-lst/selected-data

gen-rec:
	python scripts/im2rec.py --pack-label --color 1 --num-thread 1 --prefix data/selected-lst --root data/selected

train:
	python scripts/cnocr_train.py --gpu 0 --emb_model_type $(EMB_MODEL_TYPE) --seq_model_type $(SEQ_MODEL_TYPE) --optimizer adam --epoch 50 --lr 1e-5 --train_file data/selected-lst/selected-data_train --test_file data/selected-lst/selected-data_test

evaluate:
	python scripts/cnocr_evaluate.py --model-name $(MODEL_NAME) --model-epoch 2 -v -i data/selected/test.txt --image-prefix-dir data/selected --batch-size 128 -o evaluate/$(MODEL_NAME)

predict:
	python scripts/cnocr_predict.py --model_name $(MODEL_NAME) --file examples/rand_cn1.png


.PHONY: gen-lst gen-rec train evaluate predict

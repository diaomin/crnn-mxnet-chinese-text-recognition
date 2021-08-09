# 可取值：['densenet-s']
ENCODER_NAME = densenet-s
# 可取值：['fc', 'gru', 'lstm']
DECODER_NAME = gru
MODEL_NAME = $(ENCODER_NAME)-$(DECODER_NAME)
EPOCH = 41

INDEX_DIR = data
TRAIN_CONFIG_FP = examples/train_config.json

train:
	cnocr train -m $(MODEL_NAME) --index-dir $(INDEX_DIR) --train-config-fp $(TRAIN_CONFIG_FP)

evaluate:
	python scripts/cnocr_evaluate.py --model-name $(MODEL_NAME) --model-epoch $(EPOCH) -i $(REC_DATA_ROOT_DIR)/test-part.txt --image-prefix-dir $(REC_DATA_ROOT_DIR) --batch-size 128 --gpu 1 -o evaluate/$(MODEL_NAME)-$(EPOCH)

filter:
	python scripts/filter_samples.py --sample_file $(REC_DATA_ROOT_DIR)/test-part.txt --badcases_file evaluate/$(MODEL_NAME)-$(EPOCH)/badcases.txt --distance_thrsh 2 -o $(REC_DATA_ROOT_DIR)/new.txt

predict:
	cnocr predict -m $(MODEL_NAME) -f examples/rand_cn1.png



.PHONY: train predict evaluate filter

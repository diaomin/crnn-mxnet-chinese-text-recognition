# 可取值：['densenet-s']
ENCODER_NAME = densenet-lite-136
# 可取值：['fc', 'gru', 'lstm']
DECODER_NAME = fclite
MODEL_NAME = $(ENCODER_NAME)-$(DECODER_NAME)

INDEX_DIR = data/output_normal
TRAIN_CONFIG_FP = docs/examples/train_config_gpu.json

train:
	cnocr train -m $(MODEL_NAME) --index-dir $(INDEX_DIR) --train-config-fp $(TRAIN_CONFIG_FP)

evaluate:
	cnocr evaluate -m $(MODEL_NAME) -i $(REC_DATA_ROOT_DIR)/test-part.txt --image-folder $(REC_DATA_ROOT_DIR) --batch-size 128 -c cuda:0 -o eval_results/$(MODEL_NAME)-$(EPOCH)

filter:
	python scripts/filter_samples.py --sample_file $(REC_DATA_ROOT_DIR)/test-part.txt --badcases_file evaluate/$(MODEL_NAME)-$(EPOCH)/badcases.txt --distance_thrsh 2 -o $(REC_DATA_ROOT_DIR)/new.txt

predict:
	cnocr predict -m $(MODEL_NAME) -f docs/examples/rand_cn1.png



.PHONY: train predict evaluate filter

# 可取值：['densenet-s']
ENCODER_NAME = densenet-s
# 可取值：['fc', 'gru', 'lstm']
DECODER_NAME = gru
MODEL_NAME = $(ENCODER_NAME)-$(DECODER_NAME)
EPOCH = 41

INDEX_DIR = data/test
TRAIN_CONFIG_FP = examples/train_config.json

# 训练模型
train:
	cnocr train -m $(MODEL_NAME) --index-dir $(INDEX_DIR) --train-config-fp $(TRAIN_CONFIG_FP)

# 在测试集上评估模型，所有badcases的具体信息会存放到文件夹 `evaluate/$(MODEL_NAME)` 中
evaluate:
	python scripts/cnocr_evaluate.py --model-name $(MODEL_NAME) --model-epoch 1 -v -i $(DATA_ROOT_DIR)/test.txt \
		--image-prefix-dir examples --batch-size 128 -o evaluate/$(MODEL_NAME)

predict:
	cnocr predict -m $(MODEL_NAME) -i examples/rand_cn1.png


package:
	python setup.py sdist bdist_wheel

VERSION = 2.0.1
upload:
	python -m twine upload  dist/cnocr-$(VERSION)* --verbose


.PHONY: train evaluate predict package upload

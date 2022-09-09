CATEGORIES = text,english,formula
BASE_MODEL = mobilenet_v2
ROOT_DIR = data/image-formula-text
ROOT_DATA_DIR = $(ROOT_DIR)/train_data

train:
	cnocr-clf train -c $(CATEGORIES) -i $(ROOT_DATA_DIR) --image-dir $(ROOT_DATA_DIR) \
		-t '{"crop_size": [150, 450], "resize_size": 160, "resize_max_size": 1000}' \
		-b $(BASE_MODEL) --train-config-fp $(ROOT_DIR)/train_config.json

predict:
	cnocr-clf predict -c $(CATEGORIES) -b $(BASE_MODEL) \
		-t '{"crop_size": [150, 450], "resize_size": 160, "resize_max_size": 1000}' \
		-m $(ROOT_DIR)/image-clf-epoch=015-val-accuracy-epoch=0.9394-model.ckpt \
		-i $(ROOT_DATA_DIR)/dev.tsv --image-dir $(ROOT_DATA_DIR) -o $(ROOT_DIR)/dev-preds \
		--device cpu --batch-size 16

resave:
	cnocr resave -i lightning_logs/version_4/checkpoints/image-clf-epoch=014-accuracy_epoch=1.0000.ckpt \
	-o $(ROOT_DATA_DIR)/image-clf-epoch=014-accuracy_epoch=1.0000-model.ckpt

.PHONY: train predict resave


# 图片分类工具

安装 **cnocr** 后可使用命令 **`cnocr-clf`** 训练**图片分类模型**。



## 图片分类预测

使用命令 **`cnocr-clf predict`** 预测文件夹中所有图片或者索引文件中指定的所有图片，以下是使用说明：

```bash
> cnocr-clf predict -h
Usage: cnocr-clf predict [OPTIONS]

  模型预测

Options:
  -c, --categories TEXT           分类包含的类别
  -b, --base-model-name [mobilenet_v2|densenet121|efficientnet_v2_s]
                                  使用的base模型名称
  -t, --transform-configs TEXT    configs for transforms
  -m, --model-fp TEXT             模型文件路径  [required]
  -d, --device TEXT               使用cpu还是 `cuda` 运行代码，也可指定为特定gpu，如`cuda:0`。默认为
                                  `cpu`
  -i, --index-fp TEXT             索引文件,如 train.tsv 和 dev.tsv
                                  文件；或者文件所在的目录，此时预测此文件夹下的所有图片  [required]
  --image-dir TEXT                图片所在的文件夹(如果 `--index-fp` 传入的是文件夹，此参数无效)
  --batch-size INTEGER            batch size for prediction
  --num-workers INTEGER           num_workers for DataLoader
  --pin-memory                    pin_memory for DataLoader
  -o, --out-image-dir TEXT        输出图片所在的文件夹  [required]
  --out-thresholds TEXT           仅输出预测概率值在此给定值范围内的图片
  -h, --help                      Show this message and exit.
```



具体使用也可参考文件 [data/image-formula-text/image-formula-text.Makefile](https://github.com/breezedeus/cnocr/blob/master/data/image-formula-text/image-formula-text.Makefile) 。



## 模型训练

使用命令 **`cnocr-clf train`**  训练图片分类模型，以下是使用说明：

```bash
> cnocr-clf train -h
Usage: cnocr-clf train [OPTIONS]

  训练分类模型

Options:
  -c, --categories TEXT           分类包含的类别
  -b, --base-model-name [mobilenet_v2|densenet121|efficientnet_v2_s]
                                  使用的base模型名称
  -t, --transform-configs TEXT    configs for transforms
  -i, --index-dir TEXT            索引文件所在的文件夹，会读取文件夹中的 train.tsv 和 dev.tsv 文件
                                  [required]
  --image-dir TEXT                图片所在的文件夹  [required]
  --train-config-fp TEXT          识别模型训练使用的json配置文件，参考
                                  `docs/examples/train_config.json`
                                  [required]
  -r, --resume-from-checkpoint TEXT
                                  恢复此前中断的训练状态，继续训练识别模型。所以文件中应该包含训练状态。默认为
                                  `None`
  -p, --pretrained-model-fp TEXT  导入的训练好的识别模型，作为模型初始值。优先级低于"--resume-from-
                                  checkpoint"，当传入"--resume-from-
                                  checkpoint"时，此传入失效。默认为 `None`
  -h, --help                      Show this message and exit.
```



具体使用也可参考文件 [data/image-formula-text/image-formula-text.Makefile](https://github.com/breezedeus/cnocr/blob/master/data/image-formula-text/image-formula-text.Makefile) 。

# 模型训练

自带模型基于 `500+万` 的文字图片训练而成。

## 训练命令

[命令行工具](command.md) 介绍了训练命令。使用命令 **`cnocr train`**  训练文本检测模型，以下是使用说明：

```bash
> cnocr train -h
Usage: cnocr train [OPTIONS]

  训练识别模型

Options:
  -m, --rec-model-name TEXT       识别模型名称。默认值为 `densenet_lite_136-fc`
  -i, --index-dir TEXT            索引文件所在的文件夹，会读取文件夹中的 train.tsv 和 dev.tsv 文件
                                  [required]
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

例如可以使用以下命令进行训练：

```bash
> cnocr train -m densenet_lite_136-fc --index-dir data/test --train-config-fp docs/examples/train_config.json
```

训练数据的格式见文件夹 [data/test](https://github.com/breezedeus/cnocr/blob/master/data/test) 中的 [train.tsv](https://github.com/breezedeus/cnocr/blob/master/data/test/train.tsv) 和 [dev.tsv](https://github.com/breezedeus/cnocr/blob/master/data/test/dev.tsv) 文件。

具体使用也可参考文件 [Makefile](https://github.com/breezedeus/cnocr/blob/master/Makefile) 。



## 模型精调

如果需要在已有模型的基础上精调模型，需要把训练配置中的学习率设置的较小，`lr_scheduler`的设置可参考以下：

```json
{  
  "learning_rate": 3e-5,
  "lr_scheduler": {
    "name": "cos_warmup",
    "min_lr_mult_factor": 0.01,
    "warmup_epochs": 2
}
```

> 注：需要尽量避免过度精调！



## 详细训练教程和训练过程作者答疑

[**模型训练详细教程**](https://articles.zsxq.com/id_u6b4u0wrf46e.html) 见作者的 **知识星球** [CnOCR/CnSTD私享群](https://t.zsxq.com/FEYZRJQ) ，加入私享群后作者也会尽力解答训练过程中遇到的问题。此外，私享群中作者每月提供两次免费特有数据的训练服务。**抱歉的是，私享群不是免费的。**

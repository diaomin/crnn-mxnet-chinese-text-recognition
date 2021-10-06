# 模型训练

自带模型基于 `400+万` 的文字图片训练而成。


## 训练命令

[命令行工具](command.md) 介绍了训练命令。使用命令 **`cnocr train`**  训练文本检测模型，以下是使用说明：

```bash
(venv) ➜  cnocr git:(pytorch) ✗ cnocr train -h
Usage: cnocr train [OPTIONS]

Options:
  -m, --model-name [densenet-s-fc|densenet-s-lstm|densenet-s-gru]
                                  模型名称。默认值为 densenet-s-fc
  -i, --index-dir TEXT            索引文件所在的文件夹，会读取文件夹中的 train.tsv 和 dev.tsv 文件
                                  [required]

  --train-config-fp TEXT          训练使用的json配置文件，参考 `example/train_config.json`
                                  [required]

  -r, --resume-from-checkpoint TEXT
                                  恢复此前中断的训练状态，继续训练。默认为 `None`
  -p, --pretrained-model-fp TEXT  导入的训练好的模型，作为初始模型。优先级低于"--restore-training-
                                  fp"，当传入"--restore-training-fp"时，此传入失效。默认为
                                  `None`

  -h, --help                      Show this message and exit.
```



例如可以使用以下命令进行训练：

```bash
cnocr train -m densenet-s-fc --index-dir data/test --train-config-fp docs/examples/train_config.json
```


训练数据的格式见文件夹 [data/test](https://github.com/breezedeus/cnocr/blob/master/data/test) 中的 
[train.tsv](https://github.com/breezedeus/cnocr/blob/master/data/test/train.tsv) 
和 [dev.tsv](https://github.com/breezedeus/cnocr/blob/master/data/test/dev.tsv) 文件。



具体使用也可参考文件 [Makefile](https://github.com/breezedeus/cnocr/blob/master/Makefile) 。



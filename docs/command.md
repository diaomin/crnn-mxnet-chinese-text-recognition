# 脚本使用

**cnocr** 包含了几个命令行工具，安装 **cnocr** 后即可使用。



## 预测单个文件或文件夹中所有图片

使用命令 **`cnocr predict`** 预测单个文件或文件夹中所有图片，以下是使用说明：



```bash
(venv) ➜  cnocr git:(dev) ✗ cnocr predict -h
Usage: cnocr predict [OPTIONS]

Options:
  -m, --model-name TEXT           模型名称。默认值为 densenet_lite_136-fc
  -p, --pretrained-model-fp TEXT  使用训练好的模型。默认为 `None`，表示使用系统自带的预训练模型
  -c, --context TEXT              使用cpu还是 `gpu` 运行代码，也可指定为特定gpu，如`cuda:0`。默认为
                                  `cpu`

  -i, --img-file-or-dir TEXT      输入图片的文件路径或者指定的文件夹  [required]
  -s, --single-line               是否输入图片只包含单行文字。对包含单行文字的图片，不做按行切分；否则会先对图片按行分割后
                                  再进行识别

  -h, --help                      Show this message and exit.
```



例如可以使用以下命令对图片 `docs/examples/rand_cn1.png` 进行文字识别：

```bash
cnstd predict -i docs/examples/rand_cn1.png -s
```



具体使用也可参考文件 [Makefile](https://github.com/breezedeus/cnocr/blob/master/Makefile) 。




## 模型评估

使用命令 **`cnocr evaluate`** 在指定的数据集上评估模型效果，以下是使用说明：



```bash
(venv) ➜  cnocr git:(dev) ✗ cnocr evaluate -h
Usage: cnocr evaluate [OPTIONS]

Options:
  -m, --model-name TEXT           模型名称。默认值为 densenet_lite_136-fc
  -p, --pretrained-model-fp TEXT  使用训练好的模型。默认为 `None`，表示使用系统自带的预训练模型
  -c, --context TEXT              使用cpu还是 `gpu` 运行代码，也可指定为特定gpu，如`cuda:0`。默认为
                                  `cpu`

  -i, --eval-index-fp TEXT        待评估文件所在的索引文件，格式与训练时训练集索引文件相同，每行格式为 `<图片路径>
                                  <以空格分割的labels>`

  --img-folder TEXT               图片所在文件夹，相对于索引文件中记录的图片位置  [required]
  --batch-size INTEGER            batch size. 默认值：`128`
  -o, --output-dir TEXT           存放评估结果的文件夹。默认值：`eval_results`
  -v, --verbose                   whether to print details to screen
  -h, --help                      Show this message and exit.
```



例如可以使用以下命令评估 `data/test/dev.tsv` 中指定的所有样本：

```bash
cnocr evaluate -i data/test/dev.tsv --image-folder data/images 
```



具体使用也可参考文件 [Makefile](https://github.com/breezedeus/cnocr/blob/master/Makefile) 。



## 模型训练

使用命令 **`cnocr train`**  训练文本检测模型，以下是使用说明：

```bash
(venv) ➜  cnocr git:(dev) ✗ cnocr train -h
Usage: cnocr train [OPTIONS]

Options:
  -m, --model-name TEXT           模型名称。默认值为 densenet_lite_136-fc
  -i, --index-dir TEXT            索引文件所在的文件夹，会读取文件夹中的 train.tsv 和 dev.tsv 文件
                                  [required]

  --train-config-fp TEXT          训练使用的json配置文件，参考
                                  `docs/examples/train_config.json`
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
cnocr train -m densenet_lite_136-fc --index-dir data/test --train-config-fp docs/examples/train_config.json
```



训练数据的格式见文件夹 [data/test](https://github.com/breezedeus/cnocr/blob/master/data/test) 中的 [train.tsv](https://github.com/breezedeus/cnocr/blob/master/data/test/train.tsv) 和 [dev.tsv](https://github.com/breezedeus/cnocr/blob/master/data/test/dev.tsv) 文件。



具体使用也可参考文件 [Makefile](https://github.com/breezedeus/cnocr/blob/master/Makefile) 。



## 模型转存

训练好的模型会存储训练状态，使用命令 **`cnocr resave`**  去掉与预测无关的数据，降低模型大小。

```bash
(venv) ➜  cnocr git:(pytorch) ✗ cnocr resave -h
Usage: cnocr resave [OPTIONS]

  训练好的模型会存储训练状态，使用此命令去掉预测时无关的数据，降低模型大小

Options:
  -i, --input-model-fp TEXT   输入的模型文件路径  [required]
  -o, --output-model-fp TEXT  输出的模型文件路径  [required]
  -h, --help                  Show this message and exit.
```




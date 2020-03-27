import logging
import os
import mxnet as mx
from mxnet.lr_scheduler import PolyScheduler


def _load_model(args):
    if 'load_epoch' not in args or args.load_epoch is None:
        return None, None, None
    assert args.prefix is not None
    model_prefix = args.prefix
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.load_epoch)
    logging.info('Loaded model %s-%04d.params', model_prefix, args.load_epoch)
    return sym, arg_params, aux_params


def fit(network, data_train, data_val, metrics, args, hp, data_names=None):
    if args.gpu > 0:
        contexts = [mx.context.gpu(i) for i in range(args.gpu)]
    else:
        contexts = [mx.context.cpu()]

    sym, arg_params, aux_params = _load_model(args)
    if sym is not None:
        assert sym.tojson() == network.tojson()
    if not os.path.exists(os.path.dirname(args.prefix)):
        os.makedirs(os.path.dirname(args.prefix))

    module = mx.mod.Module(
            symbol=network,
            data_names=["data"] if data_names is None else data_names,
            label_names=['label'],
            context=contexts)

    # from mxnet import nd
    # import numpy as np
    # data = nd.random.uniform(shape=(128, 1, 32, 100))
    # label = np.random.randint(1, 11, size=(128, 4))
    # module.bind(data_shapes=[('data', (128, 1, 32, 100))], label_shapes=[('label', (128, 4))])
    # # e = module.bind()
    # # f = e.forward(is_train=False)
    # module.init_params(mx.init.Xavier(factor_type="in", magnitude=2.34))
    # from ..data_utils.data_iter import SimpleBatch
    # data_all = [data]
    # label_all = [mx.nd.array(label)]
    # # print(label_all[0])
    # # data_names = ['data'] + init_state_names
    # data_names = ['data']
    # label_names = ['label']
    #
    # data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
    # module.forward(data_batch)
    # f = module.get_outputs()
    # import pdb; pdb.set_trace()

    begin_epoch = args.load_epoch if args.load_epoch else 0
    num_epoch = hp.num_epoch + begin_epoch
    lr_scheduler = PolyScheduler(base_lr=hp.learning_rate, final_lr=hp.learning_rate * 0.01,
                                 warmup_steps=begin_epoch, warmup_begin_lr=0, warmup_mode='linear',
                                 max_update=hp.num_epoch // 2 + begin_epoch, pwr=2)

    optimizer_params = {'learning_rate': hp.learning_rate,
                        #'lr_scheduler': lr_scheduler,
                        # 'momentum': hp.momentum,
                        # 'wd': 0.00001,
                        }
    module.fit(train_data=data_train,
               eval_data=data_val,
               begin_epoch=begin_epoch,
               num_epoch=num_epoch,
               # use metrics.accuracy or metrics.accuracy_lcs
               eval_metric=mx.metric.np(metrics.accuracy, allow_extra_outputs=True),
               optimizer=hp.optimizer,
               optimizer_params=optimizer_params,
               initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
               arg_params=arg_params,
               aux_params=aux_params,
               batch_end_callback=mx.callback.Speedometer(hp.batch_size, 50),
               epoch_end_callback=mx.callback.do_checkpoint(args.prefix),
               )
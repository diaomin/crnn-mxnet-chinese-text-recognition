
import logging
import os
import mxnet as mx


def _load_model(args, rank=0):
    if 'load_epoch' not in args or args.load_epoch is None:
        return (None, None, None)
    assert args.prefix is not None
    model_prefix = args.prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)


def fit(network, data_train, data_val, metrics, args, hp, data_names=None):
    if args.gpu:
        contexts = [mx.context.gpu(i) for i in range(args.gpu)]
    else:
        contexts = [mx.context.cpu(i) for i in range(args.cpu)]

    sym, arg_params, aux_params = _load_model(args)
    if sym is not None:
        assert sym.tojson() == network.tojson()

    module = mx.mod.Module(
            symbol = network,
            data_names= ["data"] if data_names is None else data_names,
            label_names=['label'],
            context=contexts)

    module.fit(train_data=data_train,
               eval_data=data_val,
               begin_epoch=args.load_epoch if args.load_epoch else 0,
               num_epoch=hp.num_epoch,
               # use metrics.accuracy or metrics.accuracy_lcs
               eval_metric=mx.metric.np(metrics.accuracy, allow_extra_outputs=True),
               optimizer='AdaDelta',
               optimizer_params={'learning_rate': hp.learning_rate,
                                 # 'momentum': hp.momentum,
                                 'wd': 0.00001,
                                 },
               initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
               arg_params=arg_params,
               aux_params=aux_params,
               batch_end_callback=mx.callback.Speedometer(hp.batch_size, 50),
               epoch_end_callback=mx.callback.do_checkpoint(args.prefix),
               )
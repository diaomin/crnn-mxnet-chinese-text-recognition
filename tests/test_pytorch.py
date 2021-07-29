import torch
# from torchvision.models.densenet import DenseNet

from cnocr.consts import ENG_LETTERS
from cnocr.models.densenet import DenseNet
from cnocr.models.crnn import CRNN


def test_densenet():
    width = 280
    img = torch.randn(4, 1, 32, width)
    net = DenseNet(32, [2, 2, 2, 2], 64)
    net.eval()
    res = net(img)
    print(res.shape)

    crnn = CRNN(net, vocab=ENG_LETTERS, lstm_features=512, rnn_units=128)
    res2 = crnn(img)
    print(res2)

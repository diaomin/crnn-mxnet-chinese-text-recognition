# coding: utf-8
import string


EMB_MODEL_TYPES = ['conv', 'conv-lite', 'densenet', 'densenet-lite']
SEQ_MODEL_TYPES = ['lstm', 'gru', 'fc']

# name: (epochs, url)
AVAILABLE_MODELS = {
    'conv-lstm': (50, ),
    'conv-lite-lstm': (45, ),
    'conv-lite-fc': (27, ),
    'densenet-lite-lstm': (45, ),
    'densenet-lite-fc': (32, ),
}

# 候选字符集合
NUMBERS = string.digits + string.punctuation
ENG_LETTERS = string.digits + string.ascii_letters + string.punctuation

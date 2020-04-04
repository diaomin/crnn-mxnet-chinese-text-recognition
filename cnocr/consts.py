# coding: utf-8
import string
from .__version__ import __version__


MODEL_BASE_URL = 'https://www.dropbox.com/s/7w8l3mk4pvkt34w/cnocr-models-v1.0.0.zip?dl=1'
MODEL_EPOCE = 20
EMB_MODEL_TYPES = ['conv', 'conv-lite', 'densenet', 'densenet-lite']
SEQ_MODEL_TYPES = ['lstm', 'gru', 'fc']

ZIP_FILE_NAME = 'cnocr-models-v{}.zip'.format(__version__)

# 候选字符集合
NUMBERS = string.digits + string.punctuation
ENG_LETTERS = string.digits + string.ascii_letters + string.punctuation

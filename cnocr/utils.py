import os
import platform
import zipfile

from mxnet.gluon.utils import download
from .consts import MODEL_BASE_URL


def data_dir_default():
    """

    :return: default data directory depending on the platform and environment variables
    """
    system = platform.system()
    if system == 'Windows':
        return os.path.join(os.environ.get('APPDATA'), 'cnocr')
    else:
        return os.path.join(os.path.expanduser("~"), '.cnocr')


def data_dir():
    """

    :return: data directory in the filesystem for storage, for example when downloading models
    """
    return os.getenv('CNOCR_HOME', data_dir_default())


def get_model_file(root=data_dir()):
    r"""Return location for the downloaded models on local file system.

    This function will download from online model zoo when model cannot be found or has mismatch.
    The root directory will be created if it doesn't exist.

    Parameters
    ----------
    root : str, default $CNOCR_HOME
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    file_name = 'cnocr-models.zip'
    root = os.path.expanduser(root)

    os.makedirs(root, exist_ok=True)

    zip_file_path = os.path.join(root, file_name)
    if not os.path.exists(zip_file_path):
        download(MODEL_BASE_URL, path=zip_file_path, overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(root)
    os.remove(zip_file_path)

    return os.path.join(root, 'models')


def read_charset(charset_fp):
    alphabet = []
    # 第0个元素是预留id，在CTC中用来分割字符。它不对应有意义的字符
    with open(charset_fp) as fp:
        for line in fp:
            alphabet.append(line.rstrip('\n'))
    print('Alphabet size: %d' % len(alphabet))
    inv_alph_dict = {_char: idx for idx, _char in enumerate(alphabet)}
    # inv_alph_dict[' '] = inv_alph_dict['<space>']  # 对应空格
    return alphabet, inv_alph_dict


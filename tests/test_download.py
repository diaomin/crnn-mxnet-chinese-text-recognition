from mxnet.gluon.utils import download


def test_download():
    url = 'https://www.dropbox.com/s/5n09nxf4x95jprk/cnocr-models-v0.1.0.zip?dl=1'
    download(url, './cnocr-models.zip', overwrite=True)
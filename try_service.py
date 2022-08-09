# coding: utf-8
# 等价于命令：
# > curl -F image=@docs/examples/huochepiao.jpeg http://0.0.0.0:8000/ocr

import requests

url = 'http://0.0.0.0:8000/ocr'


def ocr():
    image = 'docs/examples/huochepiao.jpeg'
    r = requests.post(
        url,
        files={'image': (image, open(image, 'rb'), 'image/png')},
    )
    print(r)
    print(r.status_code)
    return r.json()


print(ocr())


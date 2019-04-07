#!/usr/bin/env python3
import os
from setuptools import find_packages, setup

dir_path = os.path.dirname(os.path.realpath(__file__))

required = [
    'numpy>=1.14.0,<1.15.0',
    'pillow>=5.3.0',
    'mxnet>=1.3.1,<1.4.0',
    'gluoncv>=0.3.0,<0.4.0',
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='cnocr',
    version='0.2.0',
    description="Package for Chinese OCR, which can be used after installed without training yourself OCR model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='breezedeus',
    author_email='breezedeus@163.com',
    license='Apache 2.0',
    url='https://github.com/breezedeus/cnocr',
    platforms=["Mac", "Linux", "Windows"],
    packages=find_packages(),
    # entry_points={'cnocr_predict': ['chitchatbot=chitchatbot.cli:main'],
    #               'cnocr_train': ['chitchatbot=chitchatbot:Spec']},
    include_package_data=True,
    install_requires=required,
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ],
)

#!/usr/bin/env python3
import os
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from subprocess import check_call
dir_path = os.path.dirname(os.path.realpath(__file__))

required = [
    'numpy>=1.14.0,<1.15.0',
    'pillow>=5.3.0',
    'mxnet>=1.3.1,<1.4.0',
    'gluoncv>=0.3.0,<0.4.0',
]

setup(
    name='cnocr',
    version='0.1',
    description="Package for Chinese OCR, which can be used after installed without training yourself OCR model",
    author='breezedeus',
    author_email='breezedeus@163.com',
    license='Apache 2.0',
    url='https://github.com/breezedeus/cnocr',
    platforms=["all"],
    packages=find_packages(),
    # entry_points={'console_scripts': ['chitchatbot=chitchatbot.cli:main'],
    #               'plus.ein.botlet': ['chitchatbot=chitchatbot:ChitchatBot'],
    #               'plus.ein.botlet.parser': ['chitchatbot=chitchatbot:Spec']},
    include_package_data=True,
    install_requires=required,
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache 2.0 License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ],
)

from __future__ import print_function
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnocr.utils import data_dir, read_charset

BAD_CHARS = [5751, 5539, 5536, 5535, 5464, 4105]


def main():
    charset_fp = os.path.join(data_dir(), 'models', 'label_cn.txt')
    alphabet, inv_alph_dict = read_charset(charset_fp)
    for idx in BAD_CHARS:
        print('idx: {}, char: {}'.format(idx, alphabet[idx]))


if __name__ == '__main__':
    main()

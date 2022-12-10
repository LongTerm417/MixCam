# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 13:57
# @File    : log_config.py

import os
import sys


class HiddenPrints(object):
    def __init__(self, activated=True):
        self.activated = activated
        self.original_stdout = None

    def open(self):
        sys.stdout.close()
        sys.stdout = self.original_stdout

    def close(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')

    def __enter__(self):
        if self.activated:
            self.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.activated:
            self.open()


if __name__ == '__main__':
    pass



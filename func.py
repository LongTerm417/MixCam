# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 14:59
# @File    : func.py
import json
import os


def read_json_file(file_name):
    data = dict()
    if not os.path.exists(file_name):
        return data
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_json_file(save_info, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(save_info, f, indent=4, ensure_ascii=False)
    return


if __name__ == '__main__':
    pass


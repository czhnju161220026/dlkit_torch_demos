# encoding=utf-8

import os


def main(data, epochs):
    cwd = os.getcwd()
    script = os.path.join(cwd, 'pytorch_demos/demo7/main.py')
    dataroot = os.path.join(data, 'pt_data/NAME_DATA/names/*.txt')
    os.system('python %s %s' % (script, dataroot))

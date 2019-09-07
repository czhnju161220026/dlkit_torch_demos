# encoding=utf-8

import os


def main(data, epochs):
    cwd = os.getcwd()
    script = os.path.join(cwd, 'pytorch_demos/demo9/main.py')
    dataroot = os.path.join(data, 'pt_data/wikitext-2')
    os.system('python %s --data %s --epochs %s' % (script, dataroot,epochs))
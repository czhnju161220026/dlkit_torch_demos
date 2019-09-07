# encoding=utf-8
import os


def main(data, epochs):
    cwd = os.getcwd()
    script = os.path.join(cwd, 'pytorch_demos/demo4/main.py')
    dataroot = os.path.join(data, 'pt_data')
    os.system('python %s  --epochs %s  --dataroot %s --cuda' % (script, epochs, dataroot))

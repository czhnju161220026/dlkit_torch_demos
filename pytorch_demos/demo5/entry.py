# encoding=utf-8
import os


def main(data=None, epochs=None):
    cwd = os.getcwd()
    script = os.path.join(cwd, 'pytorch_demos/demo5/main.py')
    os.system('python %s  ' % (script))

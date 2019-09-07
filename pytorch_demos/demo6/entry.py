# encoding=utf-8
import os


def main(data, epochs):
    cwd = os.getcwd()
    script = os.path.join(cwd, 'pytorch_demos/demo6/train.py')
    dataroot = os.path.join(data, 'pt_data/traindata.pt')
    os.system('python %s  --epochs %s  --dataroot %s --outf output' % (script, epochs, dataroot))

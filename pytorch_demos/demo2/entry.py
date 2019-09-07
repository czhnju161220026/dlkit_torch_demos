# encoding=utf-8

import os


def main(data, epochs):
    cwd = os.getcwd()
    outf = os.path.join(cwd, 'output')
    script = os.path.join(cwd, 'pytorch_demos/demo2/main.py')
    dataroot = os.path.join(data, 'pt_data')
    os.system('python %s --cuda --niter %s --dataset mnist --dataroot %s --outf %s' % (script, epochs, dataroot, outf))

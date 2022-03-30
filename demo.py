from importlib import import_module
import sys
import os.path as osp
import os
sys.path.insert(0,osp.join(os.getcwd(),'Config'))
a = import_module('dml_esp')
for key,value in a.__dict__.items():
    print(key,value)
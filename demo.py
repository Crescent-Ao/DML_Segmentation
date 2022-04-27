from importlib import import_module
import sys
import os.path as osp
import os

from regex import P
from utils.config import Config

# cfg = Config.fromfile(r"/home/guoshibo/DML_Segmentation/Config/dml_esp.py")
# print(cfg)
f = open("flag.txt", 'r+', encoding="utf-8")
print(f.read().strip(),'aaa')
f.close()
f = open("flag.txt",'w+', encoding="utf-8")
f.write('1')
f.close()
f = open("flag.txt", 'r+', encoding="utf-8")
print(f.read().strip(),'bbb')

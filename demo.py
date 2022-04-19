from importlib import import_module
import sys
import os.path as osp
import os

from regex import P
from utils.config import Config

cfg = Config.fromfile(r"/home/guoshibo/DML_Segmentation/Config/dml_esp.py")
print(cfg)



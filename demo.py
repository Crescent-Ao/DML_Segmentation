from importlib import import_module
import sys
import os.path as osp
import os
from utils.config import *
a = Config.fromfile(r'/home/wa/DML_Segmentation/Config/dml_esp.py')
a.thermal
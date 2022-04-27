import imp
from importlib import import_module
import sys
import os.path as osp
import os

from regex import P
from utils.config import Config
import json

# cfg = Config.fromfile(r"/home/guoshibo/DML_Segmentation/Config/dml_esp.py")
# print(cfg)
with open("flag.json", "r") as json_file:
    json_dict = json.load(json_file)
    print(json_dict)
    print(type(json_dict))
json_dict['log'] = 0
with open("flag.json", "w") as json_file:
    json_dict = json.dump(json_dict, json_file)

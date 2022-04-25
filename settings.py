import os
from os.path import join, abspath, dirname


PROJ_DIR = join(abspath(dirname(__file__)))
DATA_DIR = join(PROJ_DIR, "data")
OUT_DIR = join(PROJ_DIR, "out")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

MAX_WORD_TOKEN_NUM = 99999


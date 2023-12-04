import sys
from os.path import dirname, abspath

model_dir = dirname(dirname(abspath(__file__)))
sys.path.insert(0, model_dir)

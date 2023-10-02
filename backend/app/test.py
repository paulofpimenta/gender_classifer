import sys
import os.path as path


model_path =  path.abspath(path.join(__file__ ,"../../../model"))
sys.path.append(model_path)
print(sys.path)
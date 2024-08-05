import os
import shutil

def copy_model():
    path = shutil.copytree("../third_party/dgl/examples/pytorch/stgcn_wave", "./src/stgcn_wave", dirs_exist_ok=True)
    print("Model copied to", path, "successfully.")
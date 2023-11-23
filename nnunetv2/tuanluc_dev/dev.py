from nnunetv2.tuanluc_dev.get_network_from_plans_dev import get_encoder
from torchsummary import summary
from pprint import pprint
from nnunetv2.tuanluc_dev.utils import *
import numpy as np
from pathlib import Path

if __name__ == '__main__':
    pass       
    # root_dir = Path("/home/dtpthao/workspace/nnUNet/env/raw/Dataset032_BraTS2018_Nicegan")
    # train_dir = root_dir / "imagesTr"
    # test_dir = root_dir / "imagesTs"
    
    # for file in test_dir.iterdir():
    #     filename = file.stem.split('.')[0]
    #     if file.is_file() and (filename.endswith("0001") or filename.endswith("0003")):
    #         file.unlink()
    #         print(f"Deleted {file}")
            # break
# config parameters for UA_speech dataset

import os
import torch
from easydict import EasyDict

cfg = EasyDict()

cfg.experiment_id = 'exp_108'              # name experiment 
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
cfg.src_net = 'resnet50'            # model
cfg.augm = True
cfg.meta_file = '/home/deepfake/DysarthricGAN/Dys_classification/prepare_data/dysarthric_metadata_spec.csv'
cfg.data_root = '/home/deepfake/DysarthricGAN/Dys_classification/Spec_data'    # root directory dataset
cfg.train_pat = ["F02", "F03", "F04", "F05", "M01", "M04", "M05", "M07", "M08", "M09", "M10"]  # training patients
cfg.val_pat = []  # validation patients
cfg.test_pat = ["M11", "M12", "M14", "M16"] # test patients
cfg.lbl_threshold = 50               # threshold for binary classification
cfg.norm = 'z'                 # normalization type: 'minmax' or 'z' or none
cfg.log_dir = '/home/deepfake/DysarthricGAN/Dys_classification/main_experiment/logs'                    # path output directory 
cfg.batch_size = 64
cfg.real_mean = 0.13                       # UA-speech mean
cfg.real_std = 1.37                      # UA-speech std dev
cfg.global_mean = 0.11                   # real+gen mean
cfg.global_std = 1.15                # real+gen std dev
cfg.lr = 5e-4                       # initial learning rate
cfg.epochs = 100

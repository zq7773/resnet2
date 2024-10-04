# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from torch import nn
# import torch.optim as optim
# import torchvision
# #pip install torchvision
# from torchvision import transforms, models, datasets
# #https://pytorch.org/docs/stable/torchvision/index.html
# import imageio
# import time
# import warnings
# import random
# import sys
# import copy
# import json
# from PIL import Image
#
# with open('./CIR_images1CIR_images1/CIR_labels1.txt') as f:
#     samples = [x.strip().split(' ') for x in f.readlines()]
#     print(samples)
#
# def load_annotations(CIR_images1):
#     data_infos = {}
#     with open(CIR_images1) as f:
#         samples = [x.strip().split(' ') for x in f.readlines()]
#         for filename, gt_label in samples:
#             data_infos[filename] = np.array(gt_label, dtype=np.int64)
#     return data_infos
#
# img_label = load_annotations('./CIR_images1CIR_images1/CIR_labels1.txt')
#
# image_name = list(img_label.keys())
# label = list(img_label.values())
#
# data_dir = './CIR_images1/'
# train_dir = data_dir + '/train_filelist'
# valid_dir = data_dir + '/val_filelist'
#
# image_path = [os.path.join(train_dir,img) for img in image_name]

# coding:utf-8
from utils import *
import torch
from PIL import Image
import numpy as np
import glob
import os
import cv2
from torch.utils.data import DataLoader
from torch.utils import data
import torchvision.transforms as T
from PIL import Image
import torchvision.transforms.functional as F
labels_root = './datasets/train/labels/'
inputs_root = './datasets/train/inputs/'
labels_root = sorted(glob.glob(labels_root + '/*'))  # 并不是从大到小

#print(labels_root[2])  # ./datasets/train/labels/01-40X-0.42UM
inputs_root = sorted(glob.glob(inputs_root + '/*'))
# transforms = transforms
# labels_patches, inputs_patches = flexible_data_augmentation(labels_root, inputs_root, 32,
#                                                                       64, 16)
# print(sum(labels_patches))
# transform 相关操作放在这里，输出的是五维，输入的是三维
sample_num = 6
PATCH_SIZE = 64
STRIDE = 16
labels_matrix=[]
inputs_matrix = []
numpats = 0
for i in range(sample_num):
    print(i)
    labels_sample = sorted(glob.glob(labels_root[i] + '/*'))
    inputs_sample = sorted(glob.glob(inputs_root[i] + '/*'))
    im_label = Image.open(labels_sample[0])
    im_input = Image.open(inputs_sample[0])
    for w in range(0, 1024 - PATCH_SIZE + STRIDE, STRIDE):
        for h in range(0, 1024 - PATCH_SIZE + STRIDE, STRIDE):
            patch_label =F.crop(im_label,w, h, PATCH_SIZE,  PATCH_SIZE)
            patch_input =F.crop(im_input, w, h, PATCH_SIZE, PATCH_SIZE)   # 先列后行的顺序

            # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(patch_label)
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(patch_input)

            if np.mean(patch_input) >20:  # threshold
                numpats = numpats + 1
                labels_matrix.append(np.zeros((16, 3, PATCH_SIZE, PATCH_SIZE)))# 一种数据加强
                labels_matrix.append(np.zeros((16, 3, PATCH_SIZE, PATCH_SIZE)))

                inputs_matrix.append(np.zeros((16, 3, PATCH_SIZE, PATCH_SIZE)))
                inputs_matrix.append(np.zeros((16, 3, PATCH_SIZE, PATCH_SIZE)))

                patch_label_vflip = F.vflip(patch_label)
                patch_input_vflip = F.vflip(patch_input)



                patch_label = T.ToTensor()(patch_label)
                patch_input = T.ToTensor()(patch_input)
                patch_label_vflip = T.ToTensor()(patch_label_vflip)
                patch_input_vflip = T.ToTensor()(patch_input_vflip)


                labels_matrix[2*(numpats-1)][0:0 + 1, :, :, :]= patch_label
                labels_matrix[2*numpats - 1][0:0 + 1, :, :, :] = patch_label_vflip


                inputs_matrix[2*(numpats-1)][0:0 + 1, :, :, :] = patch_input
                inputs_matrix[2*numpats - 1][0:0 + 1, :, :, :] = patch_input_vflip

                for j in range(1, 16):
                    im_label = Image.open(labels_sample[j])
                    im_input = Image.open(inputs_sample[j])

                    patch_label = F.crop(im_label, w, h, PATCH_SIZE, PATCH_SIZE)
                    patch_input = F.crop(im_input, w, h, PATCH_SIZE, PATCH_SIZE)
                    patch_label_vflip = F.vflip(patch_label)
                    patch_input_vflip = F.vflip(patch_input)
                    # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(patch_label)
                    # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(patch_input)

                    patch_label = T.ToTensor()(patch_label)
                    patch_input = T.ToTensor()(patch_input)
                    patch_label_vflip = T.ToTensor()(patch_label_vflip)
                    patch_input_vflip = T.ToTensor()(patch_input_vflip)
                    labels_matrix[2*(numpats-1)][j:j + 1, :, :, :] = patch_label
                    labels_matrix[2*numpats - 1][j:j + 1, :, :, :] = patch_label_vflip
                    inputs_matrix[2*(numpats-1)][j:j + 1, :, :, :] = patch_input
                    inputs_matrix[2*numpats - 1][j:j + 1, :, :, :] = patch_input_vflip


labels_5d_patches = np.zeros((2*numpats, 16, 3, PATCH_SIZE, PATCH_SIZE))
inputs_5d_patches = np.zeros((2*numpats, 16, 3, PATCH_SIZE, PATCH_SIZE))

for i in range(numpats):
    labels_5d_patches[i:i+1, :, :, :, :] = labels_matrix[i]
    inputs_5d_patches[i:i+1, :, :, :, :] = inputs_matrix[i]


labels_5d_patches = labels_5d_patches.transpose(0, 2, 1, 3, 4)
inputs_5d_patches = inputs_5d_patches.transpose(0, 2, 1, 3, 4)
np.save("./dataset/label_threhold20_sample6_vflip_64patch",labels_5d_patches)
np.save("./dataset/input_threhold20_sample6_vflip_64patch", inputs_5d_patches)
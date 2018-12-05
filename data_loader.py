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
class Data_loader(data.Dataset):
    def __init__(self):
        self.labels_5d_patches=np.load("./dataset/label_threhold16_sample5_2au_64patch.npy")
        self.inputs_5d_patches = np.load("./dataset/input_threhold16_sample5_2au_64patch.npy")
    def __getitem__(self, index):
        # 1024*1024*16->16*64*64*3*
        # print('labels_patches.shape; ', labels_patches.shape)#(3616, 16, 64, 64, 3)
        # print('inputs_patches.shape; ', inputs_patches.shape)

        return self.labels_5d_patches[index], self.inputs_5d_patches[index]

    def __len__(self):
        return self.labels_5d_patches.shape[0]


def load_test_data(args):
    test_data2 = load_3D_images(glob.glob('./datasets/{}/*.tif'.format(args.test_input)))
    test_data1 = load_3D_images(glob.glob('./datasets/{}/*.tif'.format(args.test_labels)))
    return {
        'test_data1': test_data1, 'test_data2': test_data2
    }





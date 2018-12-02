from utils import *
import torch
from PIL import Image
import numpy as np
import glob
import os
import cv2
from torch.utils.data import DataLoader
from torch.utils import data
import torchvision.transforms as transforms



class Data_loader(data.Dataset):
    def __init__(self, labels_root, inputs_root, transforms=None):
        '''
        :param labels_root: confocal #labels_root = './dataset/train/labels/01-40X-0.42UM'
        :param inputs_root: widefield #inputs_root = './dataset/train/inputs/01'
        :param transforms:
        '''
        self.labels_root = sorted(glob.glob(labels_root+'/*.tif'))
        self.inputs_root = sorted(glob.glob(inputs_root+'/*.tif'))
        self.transforms = transforms
        self.labels_patches, self.inputs_patches = flexible_data_augmentation(self.labels_root, self.inputs_root, 32, 64, 16)

        # print(sum(self.labels_patches))
    def __getitem__(self, index):
        #1024*1024*16->16*64*64*3*
        #print('labels_patches.shape; ', self.labels_patches.shape)#(3616, 16, 64, 64, 3)
        #print('inputs_patches.shape; ', self.inputs_patches.shape)
        return self.labels_patches[index], self.inputs_patches[index]

    def __len__(self):
        return self.labels_patches.shape[0]


def load_test_data(args):
    test_data2 = load_3D_images(glob.glob('./datasets/{}/*.tif'.format(args.test_input)))
    test_data1 = load_3D_images(glob.glob('./datasets/{}/*.tif'.format(args.test_labels)))
    return {
        'test_data1': test_data1, 'test_data2': test_data2
    }








def flexible_data_augmentation(labels_root, inputs_root, BATCH_SIZE=32, PATCH_SIZE=64, STRIDE=16):
    #parameters
    h = 1024
    w = 1024
    c = 3
    #scales = [1.0, 0.9, 0.8, 0.7]
    scales = [1.0]
    # calculate the number of patches
    count = 0
    for s in range(len(scales)):
        new_h = int(h*scales[s])
        new_w = int(w*scales[s])
        for x in range(0,new_h-PATCH_SIZE,STRIDE):
            for y in range(0,new_w-PATCH_SIZE,STRIDE):
                count += 1
    if count % BATCH_SIZE != 0:
        numpats = (count // BATCH_SIZE + 1) * BATCH_SIZE
    else:
        numpats = count
    print('count = %d,total patches = %d,batch_size = %d,total batches = %d' % (count, numpats, BATCH_SIZE, numpats/BATCH_SIZE))

    #generate patches
    labels_patches = np.zeros((numpats, 16, PATCH_SIZE, PATCH_SIZE, c))
    inputs_patches = np.zeros((numpats, 16, PATCH_SIZE, PATCH_SIZE, c))
    labels_matrixs = [None]*len(scales)
    inputs_matrixs = [None]*len(scales)
    for s in range(len(scales)):
        new_h = int(h * scales[s])
        new_w = int(w * scales[s])
        labels_matrixs[s] = np.zeros((16, new_h, new_w, c))
        inputs_matrixs[s] = np.zeros((16, new_h, new_w, c))
        for i in range(16):
            labels_layers = cv2.imread(labels_root[i])
            inputs_layers = cv2.imread(inputs_root[i])
            labels_layers_resized = labels_layers[int((h-new_h)/2):int((h-new_h)/2+new_h), int((w-new_w)/2):int((w-new_w)/2+new_w), :]
            inputs_layers_resized = inputs_layers[int((h-new_h)/2):int((h-new_h)/2+new_h), int((w-new_w)/2):int((w-new_w)/2+new_w), :]
            labels_matrixs[s][i:i + 1, :, :, :] = labels_layers_resized
            inputs_matrixs[s][i:i + 1, :, :, :] = inputs_layers_resized


        count = 0
        for x in range(0, new_h-PATCH_SIZE, STRIDE):
            for y in range(0, new_w-PATCH_SIZE, STRIDE):
                labels_patches[count, :, :, :, :] = labels_matrixs[s][:, x:x + PATCH_SIZE, y:y + PATCH_SIZE, :]
                inputs_patches[count, :, :, :, :] = inputs_matrixs[s][:, x:x + PATCH_SIZE, y:y + PATCH_SIZE, :]
                count += 1
    if count % BATCH_SIZE != 0:
        to_pad = numpats-count
        labels_patches[-to_pad:,:,:,:,:] = labels_patches[:to_pad,:,:,:,:]
        inputs_patches[-to_pad:,:,:,:,:] = inputs_patches[:to_pad,:,:,:,:]
    inputs_patches = inputs_patches.transpose(0, 4, 1, 2, 3)/255.0
    labels_patches = labels_patches.transpose(0, 4, 1, 2, 3)/255.0

    return labels_patches, inputs_patches

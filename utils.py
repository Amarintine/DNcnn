import gc
import os
import sys
import glob
import numpy as np
from PIL import Image
import cv2
class train_data():

    def __init__(self, filepath='./data/image_clean_pat.npy'):
        self.filepath = filepath
        assert '.npy' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = np.load(self.filepath)
        print("[*] Load successfully...")
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print("In __exit__()")

def load_data(filepath='./data/image_clean_pat.npy'):
    print('data_loader')
    return train_data(filepath=filepath)





def load_3D_images(filelist):

    # pixel value range 0-255
    data = np.zeros((len(filelist),1024,1024),dtype='uint16')
    for i in range(len(filelist)):
      im = Image.open(filelist[i])
      im_array = np.array(im,dtype='uint16')
      data[i:i+1,:,:] = im_array
    data_reshaped = np.reshape(data,(1,len(filelist),1024,1024,1))
    return data_reshaped

def save_images(filepath,ground_truth, noisy_image=None, clean_image=None):#shape of ground_truth=(1,11,1024,1024,1)

    # assert the pixel value range is 0-255

    if not clean_image.any():
        clean_image = ground_truth
    ground_truth = np.reshape(ground_truth,(ground_truth.shape[1],ground_truth.shape[2],ground_truth.shape[3]))#(11,1024,1024)
    noisy_image = np.reshape(noisy_image, (noisy_image.shape[1], noisy_image.shape[2], noisy_image.shape[3]))
    clean_image = np.reshape(clean_image, (clean_image.shape[1], clean_image.shape[2], clean_image.shape[3]))
    ground_truth_layers = [None]*ground_truth.shape[0]
    noisy_image_layers = [None]*ground_truth.shape[0]
    clean_image_layers = [None]*ground_truth.shape[0]
    cat_image_layers = [None] * ground_truth.shape[0]
    for i in range(ground_truth.shape[0]):
        ground_truth_layers[i] = np.reshape(ground_truth[i:i+1,:,:],(ground_truth.shape[1],ground_truth.shape[2]))
        noisy_image_layers[i] = np.reshape(noisy_image[i:i+1,:,:],(noisy_image.shape[1],noisy_image.shape[2]))
        clean_image_layers[i] = np.reshape(clean_image[i:i+1,:,:],(clean_image.shape[1],clean_image.shape[2]))
        cat_image_layers[i] = np.concatenate([ground_truth_layers[i], noisy_image_layers[i], clean_image_layers[i]], axis=1)
        cv2.imwrite(filepath,cat_image_layers[i])

#  # set requies_grad=Fasle to avoid computation

def set_requires_grad(nets, requires_grad=False):

    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

import gc
import os
import sys
import numpy as np
from PIL import Image
import cv2


def load_3D_images(filelist):

    # pixel value range 0-255
    data = np.zeros((len(filelist),1024,1024,3),dtype='uint16')
    for i in range(len(filelist)):
      im = Image.open(filelist[i])
      im_array = np.array(im,dtype='uint16')
      data[i:i+1,:,:,:] = im_array
    data_reshaped = np.reshape(data,(1,3,len(filelist),1024,1024))
    return data_reshaped




#  # set requies_grad=Fasle to avoid computation

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

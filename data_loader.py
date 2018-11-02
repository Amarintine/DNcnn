# encoding: utf-8
from utils import *
import torch
from PIL import Image
import numpy as np
import glob
import os
scales = [0.5]  #1,0.7,0.9,
STRIDE = 10
PAT_SIZE = 40
BAT_SIZE = 64
class Data_loader():

    def __init__(self):
        pass

    def load_data(self,args):

        files1, files2 = os.listdir('.//datasets//train/input'), os.listdir('./datasets/train/labels')
        data = []
        for filepath1 in files1:
            filepath1 = glob.glob('./datasets/train/input/' + filepath1 + '/*.tif')
            data.append(self.generate_3D_patches(filepath1))
        for filepath2 in files2:
            filepath2 = glob.glob('./datasets/train/labels/' + filepath2 + '/*.tif')
            data.append(self.generate_3D_patches(filepath2))

        # normalization [0,1]
        self.data1 = data[0].astype(np.float32) / 65535.0  # [9856,1,30,40,40]  create——data batch=64
        self.data2 = data[1].astype(np.float32) / 65535.0
        self.data3 = data[2].astype(np.float32) / 65535.0
        self.data4 = data[3].astype(np.float32) / 65535.0
        self.data5 = data[4].astype(np.float32) / 65535.0
        self.data6 = data[5].astype(np.float32) / 65535.0

        print("random load datasets")
        randnum = np.random.randint(0, 100)
        np.random.seed(randnum)
        np.random.shuffle(self.data1)
        np.random.seed(randnum)
        np.random.shuffle(self.data2)
        np.random.seed(randnum)
        np.random.shuffle(self.data3)
        np.random.seed(randnum)
        np.random.shuffle(self.data4)
        np.random.seed(randnum)
        np.random.shuffle(self.data5)
        np.random.seed(randnum)
        np.random.shuffle(self.data6)


        self.data1 = torch.from_numpy(self.data1)
        self.data2 = torch.from_numpy(self.data2)
        self.data3 = torch.from_numpy(self.data3)
        self.data4 = torch.from_numpy(self.data4)
        self.data5 = torch.from_numpy(self.data5)
        self.data6 = torch.from_numpy(self.data6)

        self.clean_data=torch.cat((self.data4,self.data5,self.data6), 0)
        self.noisy_data=torch.cat((self.data1, self.data2, self.data3), 0)

        numBatch = int(self.clean_data.shape[0] / args.batch_size)
        print(numBatch)

        return numBatch

    def load_test_data(self,args):
        self.test_data1 = load_3D_images(glob.glob('./datasets/test/{}/*.tif'.format(args.test_labels)))
        self.test_data2 = load_3D_images(glob.glob('./datasets/test/{}/*.tif'.format(args.test_input)))

    def set_data(self,batch_id,batch_size):

            # [64,1,30,40,40,]
        clean_images = self.clean_data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :, :]
        noisy_images = self.noisy_data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :, :]  #打乱
        return {'clean_images': clean_images, 'noisy_images': noisy_images}

    def set_test_data(self):
        return {
            'test_data1': self.test_data1, 'test_data2': self.test_data2,
            'eval_data1': self.eval_data1, 'eval_data2': self.eval_data2
        }

    def generate_3D_patches(self, filepath):
        # calculate the number of patches
        im = Image.open(filepath[0])
        count = 0
        for s in range(len(scales)):
            newsize = (int(im.size[0] * scales[s]), int(im.size[1] * scales[s]))
            im_resized = im.resize(newsize)
            im_h, im_w = im_resized.size
            for x in range(0, im_h - PAT_SIZE, STRIDE):
                for y in range(0, im_w - PAT_SIZE, STRIDE):
                    count += 1
        print(count)
        if count % BAT_SIZE != 0:
            numpats = (count // BAT_SIZE + 1) * BAT_SIZE
        else:
            numpats = count
        print('total patches = %d,batch_size = %d,total batches = %d' % (numpats, BAT_SIZE, numpats / BAT_SIZE))

        # generate 3D matrixs
        input_3Ds = [None]  # , None, None, None
        for s in range(len(scales)):
            input_3Ds[s] = np.zeros((len(filepath), int(im.size[0] * scales[s]), int(im.size[1] * scales[s])),
                                    dtype='uint16')  # 1024*1024*11
            for i in range(len(filepath)):
                im = Image.open(filepath[i])
                newsize = (int(im.size[0] * scales[s]), int(im.size[1] * scales[s]))
                im_resized = im.resize(newsize)
                im_array = np.array(im_resized, dtype='uint16')
                input_3Ds[s][i:i + 1, :, :] = im_array

        # generate_patches
        inputs = np.zeros((int(numpats), 1, len(filepath), PAT_SIZE, PAT_SIZE), dtype='uint16')
        count = 0
        for input_3D in input_3Ds:
            input_3D = np.reshape(input_3D, (1,input_3D.shape[0], input_3D.shape[1], input_3D.shape[2]))
            for x in range(0, input_3D.shape[1] - PAT_SIZE, STRIDE):
                for y in range(0, input_3D.shape[2] - PAT_SIZE, STRIDE):
                    inputs[count, :, :, :, :] = input_3D[:,  :,x:x + PAT_SIZE, y:y + PAT_SIZE]
                    count += 1
        print(count)
        # pad the batch
        if count < numpats:
            to_pad = numpats - count
            c = inputs[:int(to_pad), :, :, :, :]
            # inputs[count:(count+int(to_pad)), :, :, :,:] = inputs[:int(to_pad), :, :, :,:]
            inputs[-int(to_pad):, :, :, :, :] = inputs[:int(to_pad), :, :, :, :]
        print("size of inputs tensor = " + str(inputs.shape))
        return inputs
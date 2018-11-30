from utils import *
import torch
from PIL import Image
import numpy as np
import glob
import os
scales = [1,0.5]  #1,0.9,0.8,0.7
STRIDE = 16
PAT_SIZE = 64
BAT_SIZE = 64
class Data_loader():

    def __init__(self):
        pass

    def load_data(self,args):

        print('preparing patches..')
        files1, files2 = sorted(os.listdir('.//datasets//train/inputs')), sorted(os.listdir('./datasets/train/labels'))
        data = []
        for filepath1 in files1:
            filepath1 = sorted(glob.glob('./datasets/train/inputs/' + filepath1 + '/*.tif'))
            data.append(self.generate_3D_patches(filepath1))
        for filepath2 in files2:
            filepath2 = sorted(glob.glob('./datasets/train/labels/' + filepath2 + '/*.tif'))
            data.append(self.generate_3D_patches(filepath2))

        # normalization [0,1]
        self.data1 = data[0].astype(np.float32) / 255.0  # [9856,1,30,40,40]  create——data batch=64
        self.data2 = data[1].astype(np.float32) / 255.0
        # self.data3 = data[2].astype(np.float32) / 255.0
        # self.data4 = data[3].astype(np.float32) / 255.0
        # self.data5 = data[4].astype(np.float32) / 255.0
        # self.data6 = data[5].astype(np.float32) / 255.0
        # self.data7 = data[6].astype(np.float32) / 255.0
        # self.data8 = data[7].astype(np.float32) / 255.0
        # self.data9 = data[8].astype(np.float32) / 255.0
        # self.data10= data[9].astype(np.float32) / 255.0

        print("random load datasets")
        randnum = np.random.randint(0, 100)
        np.random.seed(randnum);np.random.shuffle(self.data1)
        np.random.seed(randnum);np.random.shuffle(self.data2)
        # np.random.seed(randnum);np.random.shuffle(self.data3)
        # np.random.seed(randnum);np.random.shuffle(self.data4)
        # np.random.seed(randnum);np.random.shuffle(self.data5)
        # np.random.seed(randnum);np.random.shuffle(self.data6)
        # np.random.seed(randnum);np.random.shuffle(self.data7)
        # np.random.seed(randnum);np.random.shuffle(self.data8)
        # np.random.seed(randnum);np.random.shuffle(self.data9)
        # np.random.seed(randnum);np.random.shuffle(self.data10)



        self.data1 = torch.from_numpy(self.data1)
        self.data2 = torch.from_numpy(self.data2)
        # self.data3 = torch.from_numpy(self.data3)
        # self.data4 = torch.from_numpy(self.data4)
        # self.data5 = torch.from_numpy(self.data5)
        # self.data6 = torch.from_numpy(self.data6)
        # self.data7 = torch.from_numpy(self.data7)
        # self.data8 = torch.from_numpy(self.data8)
        # self.data9 = torch.from_numpy(self.data9)
        # self.data10= torch.from_numpy(self.data10)

        # self.confocal_data=torch.cat((self.data6, self.data7, self.data8,self.data9, self.data10), 0)
        # self.widefield_data=torch.cat((self.data1, self.data2, self.data3,self.data4, self.data5), 0)
        self.confocal_data = self.data2
        self.widefield_data = self.data1

        numBatch = int(self.confocal_data.shape[0] / args.batch_size)
        print(numBatch)

        return numBatch

    def load_test_data(self,args):
        self.test_data2 = load_3D_images(glob.glob('./datasets/{}/*.tif'.format(args.test_input)))
        self.test_data1 = load_3D_images(glob.glob('./datasets/{}/*.tif'.format(args.test_labels)))


    def set_data(self,batch_id,batch_size):

            # [64,1,30,40,40,]
        confocal_images = self.confocal_data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :, :]
        widefield_images = self.widefield_data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :, :]  #打乱
        return {'confocal_images': confocal_images, 'widefield_images': widefield_images}

    def set_test_data(self):
        return {
            'test_data1': self.test_data1, 'test_data2': self.test_data2
        }


    def generate_3D_patches(self, filepath):
        # calculate the number of patches

        img_example = cv2.imread(filepath[0])  # ndarray
        (h, w, c) = img_example.shape
        count = 0
        inputs = [None, None]

        for s in range(len(scales)):
            new_h = int(h * scales[s])
            new_w = int(w * scales[s])
            new_c = c
            new_size = (new_h, new_w, new_c)



            for x in range(0, new_h - PAT_SIZE, STRIDE):
                for y in range(0, new_w - PAT_SIZE, STRIDE):
                    count += 1

        if count % BAT_SIZE != 0:
            numpats = (count // BAT_SIZE + 1) * BAT_SIZE
        else:
            numpats = count
        print('count = %d,total patches = %d,batch_size = %d,total batches = %d' % (count,numpats, BAT_SIZE, numpats / BAT_SIZE))
        count = 0
        labels_patches = np.zeros((numpats, 16, PAT_SIZE, PAT_SIZE, 3))
        inputs_patches = np.zeros((numpats, 16, PAT_SIZE, PAT_SIZE, 3))
        for s in range(len(scales)):
            new_h = int(h * scales[s])
            new_w = int(w * scales[s])
            new_c = c
            new_size = (new_h, new_w, new_c)
            # print(new_size)
            # one folder's images->3D matrixs

            inputs[s] = np.zeros((16, int(h * scales[s]), int(w * scales[s]), c))  # 16*1024*1024*3
            for i in range(16):
                inputs_layers =  cv2.imread(filepath[i])
                inputs_layers_resized =  inputs_layers[0:new_h,0:new_w,0:new_c]
                inputs[s][i:i + 1, :, :, :] = inputs_layers_resized
            # generate_patches
            for i in range(0, new_h - PAT_SIZE, STRIDE):
                for j in range(0, new_w - PAT_SIZE, STRIDE):
                    
                    inputs_patches[count：count + 1, :, :, :, :] = inputs[s][:,i:i + PAT_SIZE,j:j + PAT_SIZE, :]
                    count += 1

        if count % BAT_SIZE != 0:
            to_pad = numpats - count
            inputs_patches[-int(to_pad):, :, :, :, :] = inputs_patches[:int(to_pad),:, :, :, :]

        # np.transpose(inputs_patches, (0, 4, 1, 2, 3))
        # inputs_patches.transpose( (0, 4, 1, 2, 3) )
        inputs_patches = inputs_patches.transpose(0, 4, 1, 2, 3)

        return inputs_patches

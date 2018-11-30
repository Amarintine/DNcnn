# encoding: utf-8
import torch.nn as nn
import torch
import numpy as np
class Dncnn(nn.Module):
    def __init__(self):
        super(Dncnn, self).__init__()
        # input=input.transpose(0,4,1,2,3)
        model=[nn.Conv3d(3,16,kernel_size=(16,3,3),stride=1,padding=(13,1,1)),nn.ReLU(True)]
        for layers in range(2, 3 + 1):
            model+=[nn.Conv3d(16,16,kernel_size=(16,3,3),stride=1,padding=(13,1,1),bias=False),nn.BatchNorm3d(16),
                    nn.ReLU(True)]
        model+=[nn.Conv3d(16,3,kernel_size=(16,3,3),padding=(13,1,1),stride=1)]#nn.BatchNorm3d(32),
        self.model = nn.Sequential(*model)

    def forward(self, input):
        input=torch.cat((input,input),2)  # depth dimension
        output=self.model(input)
        # return input[:,:,0:16,:,:]-output[:,:,0:16,:,:]  # if there are residual
        return output[:,:,0:16,:,:]    # no residual




# def dncnn(input):

#     filter1 = torch.randn(32, 32, 3, 3, 3)
#     filter2 = torch.randn(1, 32, 3, 3, 3)
#     output=nn.functional.conv3d(input,Variable(filter1),bias=None, stride=1)
#     nn.functional.relu(output)
#     for layers in range(2, 16 + 1):
#         output = nn.functional.conv3d(input, Variable(filter1), bias=None, stride=1)
#         output = nn.functional.relu(nn.functional.batch_norm(output,training=True))
#     output = nn.functional.conv3d(output, Variable(filter2), bias=None, stride=1)
#     return input - output






















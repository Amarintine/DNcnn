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


class UnetGenerator_3d(nn.Module):

    def __init__(self, in_dim, out_dim, num_filter):
        super(UnetGenerator_3d, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        print("\n------Initiating U-Net------\n")

        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = maxpool_3d()
        self.down_2 = conv_block_2_3d(self.num_filter, self.num_filter * 2, act_fn)
        self.pool_2 = maxpool_3d()
        self.down_3 = conv_block_2_3d(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.pool_3 = maxpool_3d()

        self.bridge = conv_block_2_3d(self.num_filter * 4, self.num_filter * 8, act_fn)
        self.bridge_1=conv_block_2_3d(self.num_filter * 2, self.num_filter * 4, act_fn)

        self.trans_1 = conv_trans_block_3d(self.num_filter * 8, self.num_filter * 8, act_fn)
        self.up_1 = conv_block_2_3d(self.num_filter * 12, self.num_filter * 4, act_fn)
        self.trans_2 = conv_trans_block_3d(self.num_filter * 4, self.num_filter * 4, act_fn)
        self.up_2 = conv_block_2_3d(self.num_filter * 6, self.num_filter * 2, act_fn)
        self.trans_3 = conv_trans_block_3d(self.num_filter * 2, self.num_filter * 2, act_fn)
        self.up_3 = conv_block_2_3d(self.num_filter * 3, self.num_filter * 1, act_fn)

        self.out = conv_block_3d(self.num_filter, out_dim, act_fn)

    def forward(self, x):    #三层 1,3,d,1024,1024
        down_1 = self.down_1(x)        # 1,32,d,1024,1024
        pool_1 = self.pool_1(down_1)  # 1,32,d/2,512,512
        down_2 = self.down_2(pool_1)   # 1,64,d/2,512,512
        pool_2 = self.pool_2(down_2)  # 1,64,d/4,256,256
        down_3 = self.down_3(pool_2)   # 1,128,d/4,256,256
        pool_3 = self.pool_3(down_3)   # 1,128,d/8,128,128
    
        bridge = self.bridge(pool_3)   # 1,256,d/8,128,128
    
        trans_1 = self.trans_1(bridge)   # 1,256,d/4,256,256
        concat_1 = torch.cat([trans_1, down_3], dim=1)  # 1,384,d/4,256,256
        up_1 = self.up_1(concat_1)  #1,128,d/4,256,256
        trans_2 = self.trans_2(up_1)  # 1,128,d/2,512,512
        concat_2 = torch.cat([trans_2, down_2], dim=1)  # 1,192,d/2,512,512
        up_2 = self.up_2(concat_2)  # 1,64,d/2,512,512
        trans_3 = self.trans_3(up_2)  # 1,64,d,1024,1024
        concat_3 = torch.cat([trans_3, down_1], dim=1) # 1,96,d,1024,1024
        up_3 = self.up_3(concat_3)  # 1,32,d,1024,1024
    
        out = self.out(up_3)  # 1,3,d,1024,1024
        return out
#     def forward(self,x):  #  两层
#         down_1 = self.down_1(x)  # 1,32,d,1024,1024
#         pool_1 = self.pool_1(down_1)  # 1,32,d/2,512,512
#         down_2 = self.down_2(pool_1)  # 1,64,d/2,512,512
#         pool_2 = self.pool_2(down_2)  # 1,64,d/4,256,256

#         bridge = self.bridge_1(pool_2)  # 1,128,d/4,256,256

#         trans_2 = self.trans_2(bridge)  # 1,128,d/2,512,512
#         concat_1 = torch.cat([trans_2, down_2], dim=1)  # 1,192,d/2,512,512
#         up_2 = self.up_2(concat_1)  # 1,64,d/2,512,512
#         trans_3 = self.trans_3(up_2)  # 1,64,d,1024,1024
#         concat_2 = torch.cat([trans_3, down_1], dim=1)  # 1,96,d,1024,1024
#         up_2 = self.up_3(concat_2)  #1,32,d,1024,1024

#         out = self.out(up_2)  # 1,3,d,1024,1024
#         return out


def conv_block_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.Conv3d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model


def conv_trans_block_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        nn.ConvTranspose3d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.BatchNorm3d(out_dim),
        act_fn,
    )
    return model


def maxpool_3d():
    pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_block_2_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block_3d(in_dim,out_dim,act_fn),
        nn.Conv3d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
    )
    return model


def conv_block_3_3d(in_dim,out_dim,act_fn):
    model = nn.Sequential(
        conv_block_3d(in_dim,out_dim,act_fn),
        conv_block_3d(out_dim,out_dim,act_fn),
        nn.Conv3d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
    )
    return model



# def loss_function(output, label):
#     batch_size, channel, x, y, z = output.size()
#     total_loss = 0
#     for j in range(x):
#         loss = 0
#         output_z = output[:, :, j:j + 1, :, :].view(batch_size,channel,y,z)
#         label_z = label[:, :, j:j + 1, :, :].view(batch_size,channel,y,z)
#
#         softmax_output_z = nn.Softmax2d()(output_z)
#         logsoftmax_output_z = torch.log(softmax_output_z)
#         label_z=label_z.type(torch.cuda.LongTensor)
#         loss = nn.NLLLoss2d()(logsoftmax_output_z, label_z)
#         total_loss += loss
#     return total_loss










































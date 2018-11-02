import torch.nn as nn
class Dncnn(nn.Module):
    def __init__(self):
        super(Dncnn, self).__init__()
        # input=input.transpose(0,4,1,2,3)
        model=[nn.Conv3d(1,32,kernel_size=3,stride=1,padding=1),nn.ReLU(True)]
        for layers in range(2, 16 + 1):
            model+=[nn.Conv3d(32,32,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm3d(32),
                    nn.ReLU(True)]
        model+=[nn.Conv3d(32,1,kernel_size=3,padding=1,stride=1)]#nn.BatchNorm3d(32),
        self.model = nn.Sequential(*model)

    def forward(self, input):
        # input_new=input.permute(0,4,1,2,3)  #[64,1,11,40,40]
        # return self.model(input_new)
        output=self.model(input)
        # output=output.permute(0, 2, 3, 4, 1)
        # return input-output  # if there are residual
        return output




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






















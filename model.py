# encoding: utf-8
import torch
from network import Dncnn,UnetGenerator_3d
from utils import set_requires_grad
import numpy as np
import cv2
from torch.autograd import Variable
import os
from torch.autograd import Variable
torch.backends.cudnn.benchmark=True
class denoiser(object):
    def __init__(self,args,input_c_dim=1,sigma=25, batch_size=64):
        self.input_c_dim = input_c_dim
        # build model
        self.batch_size=batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.net = UnetGenerator_3d().to(self.device)
        self.net=UnetGenerator_3d(in_dim = 3, out_dim = 3, num_filter = 32).to(self.device)
        self.loss = torch.nn.MSELoss(size_average=False)
        self.adjust_learning_rate(args)
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=args.lr, betas=(args.beta1, 0.999))

    def adjust_learning_rate(self,args):
        lr = args.lr * np.ones([args.epoch])
        lr[5:] = lr[0] / 10.0

    def set_input(self,data):
        self.widefield_images_all = Variable(data[1]).to(self.device).float()
        self.confocal_images_all = Variable(data[0]).to(self.device).float()
        return self.widefield_images_all,self.confocal_images_all

    def set_test_input(self,input):
        self.data_ = input['test_data1']
        self.test_data_labels = self.data_
        self.test_data_input = input['test_data2']


    def forward(self):
        self.desired = self.net(self.widefield_images_all)   # batch,3,16,64,64

    def out(self):
        return self.desired

    def loss_calculate(self):
        # self.loss_all = loss_function(self.desired, self.confocal_images_all)
        self.loss_all = self.loss(self.desired, self.confocal_images_all)

    def get_loss(self):
        return self.loss_all

    def optimize_parameters(self):
        self.forward()
        set_requires_grad(self.net, True)
        self.optimizer.zero_grad()
        self.loss_calculate()

        self.loss_all.backward()
        self.optimizer.step()

    def evaluate(self, epoch, arg):
        # assert test_data value range is 0-255
        print("[*] Evaluating...")
        self.eval_data1 = self.eval_data1.astype(np.float32) / 255.0
        self.eval_data2 = self.eval_data2.astype(np.float32) / 255.0
        confocal_image = Variable(torch.from_numpy(self.eval_data1)).to(self.device)
        widefield_image = Variable(torch.from_numpy(self.eval_data2)).to(self.device)
        with torch.no_grad():
            output_confocal_image = self.net(widefield_image)
            # output = output.permute(0,2,3,4,1)
            # output_confocal_image = widefield_image- output  # that is desired
            confocal_image = confocal_image.cpu().numpy()
            widefield_image = widefield_image.cpu().numpy()
            output_confocal_image = output_confocal_image.cpu().numpy()
            groundtruth = np.clip(255 * self.eval_data1, 0, 255).astype('uint16')  # 查看数据类型
            widefieldimage = np.clip(255 * widefield_image, 0, 255).astype('uint16')
            outputimage = np.clip(255 * output_confocal_image, 0, 255).astype('uint16')
            # 判断x对象是否为空对象，如果都为空、0、false，则返回false，如果不都为空、0、false，则返回true
            if not confocal_image.any():
                confocal_image = groundtruth
            ground_truth = np.reshape(groundtruth, (
            groundtruth.shape[2], groundtruth.shape[3], groundtruth.shape[4], 3))  # (16,1024,1024,3)
            widefield_image = np.reshape(widefieldimage, (widefieldimage.shape[2], widefieldimage.shape[3], widefieldimage.shape[4], 3))
            output_confocal_image = np.reshape(outputimage,
                                            (outputimage.shape[2], outputimage.shape[3], outputimage.shape[4], 3))
            ground_truth_layers = [None] * ground_truth.shape[0]
            widefield_image_layers = [None] * ground_truth.shape[0]
            output_confocal_image_layers = [None] * ground_truth.shape[0]
            cat_image_layers = [None] * ground_truth.shape[0]
            for i in range(ground_truth.shape[0]):
                ground_truth_layers[i] = np.reshape(ground_truth[i:i + 1, :, :,:],
                                                    (ground_truth.shape[1], ground_truth.shape[2],3))
                widefield_image_layers[i] = np.reshape(widefield_image[i:i + 1, :, :,:], (widefield_image.shape[1], widefield_image.shape[2],3))
                output_confocal_image_layers[i] = np.reshape(output_confocal_image[i:i + 1, :, :,:], (output_confocal_image.shape[1], output_confocal_image.shape[2],3))
                cat_image_layers[i] = np.concatenate([ground_truth_layers[i], widefield_image_layers[i], output_confocal_image_layers[i]],
                                                     axis=1)
                cv2.imwrite(os.path.join(arg.sample_dir, 'show_layer','%d.tif') % (i), cat_image_layers[i])
                cv2.imwrite(os.path.join(arg.sample_dir, 'label_layer','%d.tif') % (i), ground_truth_layers[i])
                cv2.imwrite(os.path.join(arg.sample_dir, 'input_layer','%d.tif') % (i), widefield_image_layers[i])
                cv2.imwrite(os.path.join(arg.sample_dir, 'denoised_layer','%d.tif') % (i), output_confocal_image_layers[i])
                cv2.imwrite(os.path.join(arg.sample_dir, 'denoised_labels_layer','%d.tif') % (i),
                            (output_confocal_image_layers[i].astype('int16') - ground_truth_layers[i].astype('int16')))

    def test(self,arg):
        print("start testing....")

        self.test_data_input = self.test_data_input.astype(np.float32)/255.0   #[1,3,16,1024,1024]
        widefield_image = Variable(torch.from_numpy(self.test_data_input)).to(self.device)  # 1,3,16,1024,1024
        self.load_networks(arg,'latest')
        with torch.no_grad():
            output_confocal_image = self.net(widefield_image)
            widefield_image = widefield_image.cpu().numpy()
            output_confocal_image = output_confocal_image.cpu().numpy()
            groundtruth = self.test_data_labels.astype('uint8')
            # widefieldimage =widefield_image.astype('uint8')
            # outputimage =output_confocal_image.astype('uint8')
            widefieldimage = np.clip(255 * widefield_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_confocal_image, 0, 255).astype('uint8')


            ground_truth = np.reshape(groundtruth, (groundtruth.shape[2], groundtruth.shape[3], groundtruth.shape[4],3))  # (16,1024,1024,3)
            widefield_image = np.reshape(widefieldimage, (widefieldimage.shape[2], widefieldimage.shape[3], widefieldimage.shape[4],3))
            output_confocal_image = np.reshape(outputimage, (outputimage.shape[2], outputimage.shape[3], outputimage.shape[4],3))

            ground_truth_layers = [None] * ground_truth.shape[0]
            widefield_image_layers = [None] * ground_truth.shape[0]
            output_confocal_image_layers = [None] * ground_truth.shape[0]
            cat_image_layers = [None] * ground_truth.shape[0]

            for i in range(ground_truth.shape[0]):
                ground_truth_layers[i] = np.reshape(ground_truth[i:i + 1, :, :,:],
                                                    (ground_truth.shape[1], ground_truth.shape[2],3))
                widefield_image_layers[i] = np.reshape(widefield_image[i:i + 1, :, :,:], (widefield_image.shape[1], widefield_image.shape[2],3))
                output_confocal_image_layers[i] = np.reshape(output_confocal_image[i:i + 1, :, :,:], (output_confocal_image.shape[1], output_confocal_image.shape[2],3))
                cat_image_layers[i] = np.concatenate([ground_truth_layers[i], widefield_image_layers[i], output_confocal_image_layers[i]],
                                                     axis=1)
                cv2.imwrite(os.path.join(arg.sample_dir, 'show_layer','%d.tif') % (i), cat_image_layers[i])
                cv2.imwrite(os.path.join(arg.sample_dir, 'label_layer','%d.tif') % (i), ground_truth_layers[i])
                cv2.imwrite(os.path.join(arg.sample_dir, 'input_layer','%d.tif') % (i), widefield_image_layers[i])
                cv2.imwrite(os.path.join(arg.sample_dir, 'denoised_layer','%d.tif') % (i), output_confocal_image_layers[i])
                cv2.imwrite(os.path.join(arg.sample_dir, 'denoised_labels_layer','%d.tif') % (i),
                            (output_confocal_image_layers[i].astype('int16') - ground_truth_layers[i].astype('int16')))
                mse = ((ground_truth_layers[i].astype(np.float) - output_confocal_image_layers[i].astype(np.float)) ** 2).mean()
                # psnr = 10 * np.log10(65535 ** 2 / mse)
                # print('psnr for layer%d is %f' % (i, psnr))
                mse = ((widefield_image_layers[i].astype(np.float) - output_confocal_image_layers[i].astype(np.float)) ** 2).mean()
                # psnr2 = 10 * np.log10(65535 ** 2 / mse)
                #
                # print('psnr for layer%d is %f' % (i, psnr2))



    def save_networks(self,args,epoch):

        checkpoint_dir = args.ckpt_dir
        self.save_path = os.path.join(checkpoint_dir, '%s_net.pth' % (epoch))
        print("[*] Saving model...") # net = getattr(self, 'net' + name)  # 返回对象属性值
        torch.save(self.net.state_dict(), self.save_path)

    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        num_params=0
        for param in self.net.parameters():
            num_params += param.numel()
        if verbose:
            print(self.net)
        print('Total number of parameters : %.3f M' % (num_params / 1e6))

    def eval(self):
        self.net.eval()

    def load_networks(self,args, epoch):

        load_filename = '%s_net.pth' % (epoch)
        load_path = os.path.join(args.ckpt_dir, load_filename)
        print('loading the model from %s' % load_path)
        # self.net.eval()
        # print(self.net)
        # for param in self.net.parameters():
        #     print(type(param.data), param.size())
        #     print(list(param.data))
        self.net.load_state_dict(torch.load(load_path))



        # print(self.net.state_dict().keys())
        # # 参数的keys
        #
        # for key in self.net.state_dict():  # 模型参数
        #     print (key, 'corresponds to', list(self.net.state_dict()[key]))














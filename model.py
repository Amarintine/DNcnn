# encoding: utf-8
import torch
from network import Dncnn
from utils import set_requires_grad
import numpy as np
import cv2
import os
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True
class denoiser(object):
    def __init__(self,args,input_c_dim=1,sigma=25, batch_size=64):
        self.input_c_dim = input_c_dim
        # build model
        self.batch_size=batch_size
        self.device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu")
        self.net = Dncnn().to(self.device)
        self.loss = torch.nn.MSELoss(size_average=False)
        self.adjust_learning_rate(args)
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=args.lr, betas=(args.beta1, 0.999))

    def adjust_learning_rate(self,args):
        lr = args.lr * np.ones([args.epoch])
        lr[3:] = lr[0] / 10.0

    def set_input(self,input):
        self.noisy_images_all =Variable(input['noisy_images']).to(self.device)
        self.clean_images_all =Variable(input['clean_images']).to(self.device)


    def set_test_input(self,input):
        self.test_data1 = input['test_data1']
        self.test_data2 = input['test_data2']
        self.eval_data1 = input['eval_data1']
        self.eval_data2 = input['eval_data2']


    def forward(self):
        self.desired = self.net(self.noisy_images_all)

    def loss_calculate(self):
        self.loss_all = self.loss(self.desired, self.clean_images_all)

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
        self.eval_data1 = self.eval_data1.astype(np.float32) / 65535.0
        self.eval_data2 = self.eval_data2.astype(np.float32) / 65535.0
        clean_image = Variable(torch.from_numpy(self.eval_data1)).to(self.device)
        noisy_image = Variable(torch.from_numpy(self.eval_data2)).to(self.device)
        with torch.no_grad():
            output_clean_image = self.net(noisy_image)
            clean_image = clean_image.cpu().numpy()  # clean image is label
            noisy_image = noisy_image.cpu().numpy()
            output_clean_image = output_clean_image.cpu().numpy()
            groundtruth = np.clip(65535 * self.eval_data1, 0, 65535).astype('uint16')
            noisyimage = np.clip(65535 * noisy_image, 0, 65535).astype('uint16')
            outputimage = np.clip(65535 * output_clean_image, 0, 65535).astype('uint16')
            if not clean_image.any():
                clean_image = groundtruth
            ground_truth = np.reshape(groundtruth, (
            groundtruth.shape[2], groundtruth.shape[3], groundtruth.shape[4]))  # (11,1024,1024)
            noisy_image = np.reshape(noisyimage, (noisyimage.shape[2], noisyimage.shape[3], noisyimage.shape[4]))
            output_clean_image = np.reshape(outputimage,
                                            (outputimage.shape[2], outputimage.shape[3], outputimage.shape[4]))

            ground_truth_layers = [None] * ground_truth.shape[0]
            noisy_image_layers = [None] * ground_truth.shape[0]
            output_clean_image_layers = [None] * ground_truth.shape[0]
            cat_image_layers = [None] * ground_truth.shape[0]

            for i in range(ground_truth.shape[0]):
                ground_truth_layers[i] = np.reshape(ground_truth[i:i + 1, :, :], (ground_truth.shape[1], ground_truth.shape[2]))
                noisy_image_layers[i] = np.reshape(noisy_image[i:i + 1, :, :], (noisy_image.shape[1], noisy_image.shape[2]))
                output_clean_image_layers[i] = np.reshape(output_clean_image[i:i + 1, :, :], (output_clean_image.shape[1], output_clean_image.shape[2]))
                mse = ((ground_truth_layers[i].astype(np.float) - output_clean_image_layers[i].astype(np.float)) ** 2).mean()
                psnr = 10 * np.log10(65535 ** 2 / mse)
                cat_image_layers[i] = np.concatenate([ground_truth_layers[i], noisy_image_layers[i], output_clean_image_layers[i]], axis=1)
                cv2.imwrite(os.path.join(arg.sample_dir, 'eval_layer','%depoch_layer%d_psnr%f.tif' % ( epoch,i,psnr)), cat_image_layers[i])

    def test(self,arg):
        print("start testing....")
        self.test_data1 = self.test_data1.astype(np.float32) / 65535.0
        self.test_data2 = self.test_data2.astype(np.float32) / 65535.0 #[1,1,7,1024,1024]
        clean_image = Variable(torch.from_numpy(self.test_data1)).to(self.device)
        noisy_image = Variable(torch.from_numpy(self.test_data2)).to(self.device)
        self.load_networks(arg,'latest')
        with torch.no_grad():

            output_clean_image = self.net(noisy_image)
            # output_clean_image = noisy_image - output  # that is desired
            clean_image = clean_image.cpu().numpy()   # clean image is label
            noisy_image = noisy_image.cpu().numpy()
            output_clean_image = output_clean_image.cpu().numpy()
            groundtruth = np.clip(65535 * self.test_data1, 0, 65535).astype('uint16')
            noisyimage = np.clip(65535 * noisy_image, 0, 65535).astype('uint16')
            outputimage = np.clip(65535 * output_clean_image, 0, 65535).astype('uint16')
            if not clean_image.any():
                clean_image = groundtruth
            ground_truth = np.reshape(groundtruth, (groundtruth.shape[2], groundtruth.shape[3], groundtruth.shape[4]))  # (7,1024,1024)
            noisy_image = np.reshape(noisyimage, (noisyimage.shape[2], noisyimage.shape[3], noisyimage.shape[4]))
            output_clean_image = np.reshape(outputimage, (outputimage.shape[2], outputimage.shape[3], outputimage.shape[4]))

            ground_truth_layers = [None] * ground_truth.shape[0]
            noisy_image_layers = [None] * ground_truth.shape[0]
            output_clean_image_layers = [None] * ground_truth.shape[0]
            cat_image_layers = [None] * ground_truth.shape[0]
            for i in range(ground_truth.shape[0]):
                ground_truth_layers[i] = np.reshape(ground_truth[i:i + 1, :, :],
                                                    (ground_truth.shape[1], ground_truth.shape[2]))
                noisy_image_layers[i] = np.reshape(noisy_image[i:i + 1, :, :], (noisy_image.shape[1], noisy_image.shape[2]))
                output_clean_image_layers[i] = np.reshape(output_clean_image[i:i + 1, :, :], (output_clean_image.shape[1], output_clean_image.shape[2]))
                cat_image_layers[i] = np.concatenate([ground_truth_layers[i], noisy_image_layers[i], output_clean_image_layers[i]],
                                                     axis=1)
                cv2.imwrite(os.path.join(arg.sample_dir, 'show_layer','%d.tif') % (i), cat_image_layers[i])
                cv2.imwrite(os.path.join(arg.sample_dir, 'label_layer','%d.tif') % (i), ground_truth_layers[i])
                cv2.imwrite(os.path.join(arg.sample_dir, 'input_layer','%d.tif') % (i), noisy_image_layers[i])
                cv2.imwrite(os.path.join(arg.sample_dir, 'denoised_layer','%d.tif') % (i), output_clean_image_layers[i])
                cv2.imwrite(os.path.join(arg.sample_dir, 'denoised_labels_layer','%d.tif') % (i),
                            (output_clean_image_layers[i].astype('int16') - ground_truth_layers[i].astype('int16')))
                mse = ((ground_truth_layers[i].astype(np.float) - output_clean_image_layers[i].astype(np.float)) ** 2).mean()
                psnr = 10 * np.log10(65535 ** 2 / mse)
                print('psnr for layer%d is %f' % (i, psnr))
                # mse = ((noisy_image_layers[i].astype(np.float) - output_clean_image_layers[i].astype(np.float)) ** 2).mean()
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
        self.net.load_state_dict(torch.load(load_path))

        # print(self.net)
        # for param in self.net.parameters():
        #     print(type(param.data), param.size())
        #     print(list(param.data))
        #
        # print(self.net.state_dict().keys())
        # # 参数的keys
        #
        # for key in self.net.state_dict():  # 模型参数
        #     print (key, 'corresponds to', list(self.net.state_dict()[key]))














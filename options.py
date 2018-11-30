# encoding: utf-8
import argparse

class Options():

    def initialize(self):
      
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch')
        parser.add_argument('--load_epoch', type=str, default='latest',help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='# images in batch')
        parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--sigma', dest='sigma', type=int, default=25, help='noise level')
        parser.add_argument('--phase', dest='phase', default='train', help='train or test')
        parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
        parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
        parser.add_argument('--eval', default=True, help='use eval mode during test time.')
        parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
        parser.add_argument('--eval_labels', dest='eval_labels', default='./eval/labels-sample8', help='dataset for eval labels in training')
        parser.add_argument('--eval_input', dest='eval_input', default='./eval/input-sample8', help='dataset for eval input in training')
        parser.add_argument('--test_input', dest='test_input', default='./test/inputs/16', help='dataset for testing')
        parser.add_argument('--test_labels', dest='test_labels', default='./test/labels/16-40X-0.42UM', help='dataset for testing')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        args = parser.parse_args()
        return args


















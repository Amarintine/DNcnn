# encoding: utf-8
from options import Options
from model import denoiser

from data_loader import Data_loader
import time

if __name__ == '__main__':

    args=Options().initialize()
    model = denoiser(args)
    if args.eval:
        model.eval()
    dataset=Data_loader()
    dataset.load_test_data(args)
    data = dataset.set_test_data()
    model.set_test_input(data)
    model.test(args)


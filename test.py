# encoding: utf-8
from options import Options
from model import denoiser

from data_loader import load_test_data
import time

if __name__ == '__main__':

    args=Options().initialize()
    model = denoiser(args)
    if args.eval:
        model.eval()
    # data = load_test_data(args)
    # model.set_test_input(data)
    model.test_new(args,64)


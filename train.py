from options import Options
from model import denoiser
from data_loader import Data_loader
import time
import numpy as np
if __name__ == '__main__':

    args=Options().initialize()
    model = denoiser(args)
    eval_every_epoch = 1
    dataset=Data_loader()
    start_epoch = 0
    start_step = 0
    numBatch=dataset.load_data(args)
    start_time = time.time()
    print("start training....")
    model.print_networks(args.verbose)
    for epoch in range(start_epoch, args.epoch):
        for batch_id in range(start_step, numBatch):
            data=dataset.set_data(batch_id,args.batch_size)
            model.set_input(data)
            model.optimize_parameters()
            loss=model.get_loss()
            print("Epoch: [%2d] [%4d/%4d]  Time Taken: %4.4f minute, loss: %.6f"
                  % (epoch + 1, batch_id + 1, numBatch, (time.time() - start_time)/ 60.0, loss))
        model.save_networks(args,epoch+1)
        model.save_networks(args,'latest')
    print("[*] Finish training.")


















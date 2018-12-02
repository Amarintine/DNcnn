# encoding: utf-8
from options import Options
from model import denoiser
from data_loader import Data_loader
import time

from tensorboardX import SummaryWriter

if __name__ == '__main__':

    args=Options().initialize()
    model = denoiser(args)
    eval_every_epoch = 1
    dataset=Data_loader()
    writer = SummaryWriter()
    start_epoch = 0
    start_step = 0
    numBatch=dataset.load_data(args)
    start_time = time.time()
    print("start training....")
    model.print_networks(args.verbose)
    niter=0
    for epoch in range(start_epoch, args.epoch):
        for batch_id in range(start_step, numBatch):
            data=dataset.set_data(batch_id,args.batch_size)
            model.set_input(data)
            # model.evaluate(iter_num, args) # whether pretrained
            model.optimize_parameters()
            loss=model.get_loss()
            niter=niter+1
            print("Epoch: [%2d] [%4d/%4d]  Time Taken: %4.4f minute, loss: %.6f"
                  % (epoch + 1, batch_id + 1, numBatch, (time.time() - start_time)/ 60.0, loss))
            writer.add_scalar('loss_value', loss, niter)

        # if np.mod(epoch + 1, eval_every_epoch) == 0:
        #     model.evaluate(epoch + 1,args)
        model.save_networks(args,epoch+1)
        model.save_networks(args,'latest')
    print("[*] Finish training.")
    writer.close()



































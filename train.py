# encoding: utf-8from options import Optionsfrom model import denoiserfrom data_loader import Data_loaderimport timefrom torch.utils.data import DataLoaderimport torchvision.utils as tvfrom tensorboardX import SummaryWriterimport numpy as npif __name__ == '__main__':    args=Options().initialize()    model = denoiser(args)    writer = SummaryWriter()    start_epoch = 0    start_time = time.time()    print("start training....")    model.print_networks(args.verbose)    niter=0    dataset = Data_loader('./datasets/train/labels/01-40X-0.42UM', './datasets/train/inputs/01')    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=False)    batch_num=len(dataloader)    for epoch in range(start_epoch, args.epoch):        batch_id=0        for labels, inputs in dataloader:            batch_id = batch_id + 1            data=[labels, inputs]            input_widefield,input_confocal= model.set_input(data)            # model.evaluate(iter_num, args) # whether pretrained            model.optimize_parameters()            loss = model.get_loss()            niter = niter + 1            print("Epoch: [%2d]   [%4d/%4d] Time Taken: %4.4f minute, loss: %.6f" % (epoch + 1, batch_id , batch_num, (time.time() - start_time) / 60.0, loss))            writer.add_scalar('loss_value', loss, niter)            writer.add_graph(model, input_widefield, True)            x1 = tv.make_grid(np.squeeze(input_widefield[:, :, 8:8 + 1, :, :]), normalize=True, scale_each=True)            writer.add_image('input_widefield', x1, niter)            x2 = tv.make_grid(np.squeeze(input_confocal[:, :, 8:8 + 1, :, :]), normalize=True, scale_each=True)            writer.add_image('input_confocal', x2, niter)            out_confocal = model.out()            y = tv.make_grid(np.squeeze(out_confocal[:, :, 0: 1, :, :]), normalize=True, scale_each=True)            writer.add_image('out_confocal', y, niter)    model.save_networks(args, epoch + 1)    model.save_networks(args, 'latest')    print("[*] Finish training.")    writer.close()
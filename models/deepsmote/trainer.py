from models.base import Trainer
from models.archs.modules import *

import os
import torch.nn as nn

from .helper import *
from utils.tools import *
from dataloader.data_utils import *


class DeepSMOTE(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        # Default
        self.args.dim_h = 64 # factor controlling size of hidden layers
        self.args.n_channel = 3 # number of channels in the input data
        self.args.n_z = 600 # number of dimensions in latent space
        self.args.sigma = 1.0 # variance in n_z
        self.args.lambda0 = 0.01 # hyper param for weight of discriminator loss
        self.args.lr = 0.0002 # learning rate for Adam optimizer .000
        self.args.epochs = args.epochs_gen # how many epochs to run for
        self.args.batch_size = 100 # batch size for SGD

        self.encoder = Encoder(args).cuda()
        self.decoder = Decoder(args).cuda()
        
        if self.args.deepsmote_generate:
            path = os.path.join(args.save_path, 'ckpts', 'encoder.pth')
            self.encoder.load_state_dict(torch.load(path), strict=False)
            path = os.path.join(args.save_path, 'ckpts', 'decoder.pth')
            self.decoder.load_state_dict(torch.load(path), strict=False)
        
        self.criterion = nn.MSELoss().cuda()
        self.save_data_path = os.path.join(args.save_path, 'upsampled_data')
        os.makedirs(self.save_data_path, exist_ok=True)

        if args.start_session > 0:
            print('WARING: Random init weights for new sessions!')

    def get_optimizer(self, lr):
        enc_optim = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        dec_optim = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        
        return enc_optim, dec_optim

    def train(self):
        if self.args.deepsmote:
            self.train_generator()
        elif self.args.deepsmote_generate:
            self.generate()

    def generate(self):
        args = self.args
        time_module = TimeCalculator()
        time_module.time_start()
        t_start_time = time.time()

        result_list = [args]
        train_set, trainloader, testloader = get_joint_dataloader(self.args, args.sessions - 1)
        dec_x, dec_y = extract_from_concatdataset(train_set)

        session_generate(self.encoder, self.decoder, dec_x, dec_y, self.save_data_path)

        time_module.time_end()
        t_end_time_gpu = time_module.return_total_sec()

        t_end_time = time.time()
        total_time = t_end_time - t_start_time

        time_msg = '\nTotal time used %.2f secs (%.2f mins)' % (total_time, total_time / 60)
        gpu_time_msg = 'Total time used %.2f (ms) with TimeCalculator' % t_end_time_gpu

        result_list.append(time_msg)
        result_list.append(gpu_time_msg)
        print(time_msg)
        print(gpu_time_msg)

        save_list_to_txt(os.path.join(args.save_path, 'results_gen.txt'), result_list)

    def train_generator(self):
        args = self.args
        time_module = TimeCalculator()
        time_module.time_start()
        t_start_time = time.time()

        result_list = [args]
        result_list = calc_gflops(result_list, self.encoder, (3, 32, 32))
        result_list = calc_gflops(result_list, self.decoder, (1, 600))

        enc_optim, dec_optim = self.get_optimizer(self.args.lr)
        train_set, trainloader, testloader = get_joint_dataloader(self.args, args.sessions-1)
        dec_x, dec_y = extract_from_concatdataset(train_set)

        for epoch in range(args.epochs):
            train_loss, tmse_loss, tdiscr_loss = session_train(
                self.encoder, self.decoder, trainloader, enc_optim, dec_optim, self.criterion, dec_x, dec_y)

            print('Epoch: {} \tTrain Loss: {:.6f} \tmse loss: {:.6f} \tmse2 loss: {:.6f}'.format(
                epoch, train_loss, tmse_loss, tdiscr_loss))
        
        # Save the last model
        save_dir = os.path.join(args.save_path, 'ckpts')
        ensure_path(save_dir)
        torch.save(dict(params=self.encoder.state_dict()), os.path.join(save_dir, 'encoder.pth'))
        torch.save(dict(params=self.decoder.state_dict()), os.path.join(save_dir, 'decoder.pth'))
        print('Saving model to :%s' % save_dir)

        time_module.time_end()
        t_end_time_gpu = time_module.return_total_sec()
        
        t_end_time = time.time()
        total_time = t_end_time - t_start_time
        
        time_msg = '\nTotal time used %.2f secs (%.2f mins)' % (total_time, total_time / 60)
        gpu_time_msg = 'Total time used %.2f (ms) with TimeCalculator' % t_end_time_gpu

        result_list.append(time_msg)
        result_list.append(gpu_time_msg)
        print(time_msg)
        print(gpu_time_msg)

        save_list_to_txt(os.path.join(args.save_path, 'results_train.txt'), result_list)

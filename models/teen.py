from models.base import Trainer
from models.archs.networks import TEENNet
from dataloader.data_utils import *

import torch.nn as nn
from copy import deepcopy

from utils.helper import *
from utils.tools import *
from utils.metrics import *


class TEEN(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)
        
        self.model = TEENNet(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARNING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def train(self):
        args = self.args
        time_module = TimeCalculator()
        time_module.time_start()
        t_start_time = time.time()
        all_acc = []

        self.result_list = calc_gflops(self.result_list, self.model, (3, 32, 32))
        
        for session in range(args.start_session, args.sessions):
            self.model.load_state_dict(self.best_model_dict)
        
            if session == 0:  # load base class train img label
                train_set, trainloader, valloader, testloader = get_base_dataloader(self.args)
                
                if not args.only_do_incre:
                    
                    print('new classes for this session:\n', np.unique(train_set.targets))
                    optimizer, scheduler = self.get_base_optimizer(args, self.model)
                    
                    for epoch in range(args.epochs_base):
                        start_time = time.time()

                        tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                        vl, va = validate(self.model, valloader, epoch, args, session)

                        # Save better model
                        if (va * 100) >= self.trlog['max_acc'][session]:
                            self.trlog['max_acc'][session] = float('%.3f' % (va * 100))
                            self.trlog['max_acc_epoch'] = epoch
                            self.save_model(self.model, session, args.save_path)
                            self.best_model_dict = deepcopy(self.model.state_dict())
                            print('********A better model is found!!**********')
                        print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                           self.trlog['max_acc'][session]))

                        lrc = scheduler.get_last_lr()[0]
                        self.log_train_info(tl, ta, epoch, lrc, vl, va)

                        print('This epoch takes %d seconds' % (time.time() - start_time),
                            '\nstill need around %.2f mins to finish this session' % (
                                    (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                        scheduler.step()
                    
                else:
                    print('>>> Load Model &&& Finish base train...')
                    assert args.model_dir is not None

                self.model.load_state_dict(self.best_model_dict)
                if not args.not_data_init:
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    self.save_model(self.model, session, args.save_path)
                    print('Replace the fc with average embedding...')
                    self.best_model_dict = deepcopy(self.model.state_dict())

                    self.model.module.mode = 'avg_cos'
                
                acc, _, acc_per_task_list = test(self.model, testloader, args, session)
                all_acc.append(acc_per_task_list)
                g_acc, g_auc = get_gacc(ratio=int(args.base_class / args.way), all_acc=all_acc)
                
                self.log_session_info(session, acc, g_acc, g_auc)
            
            # incremental learning sessions
            else:  
                print("training session: [%d]" % session)
                train_set, trainloader, testloader = get_new_dataloader(self.args, session)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                
                if args.soft_mode == 'soft_proto':
                    self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)
                    self.model.module.soft_calibration(args, session)
                else:
                    raise NotImplementedError
                
                acc, _, acc_per_task_list = test(self.model, testloader, args, session)
                all_acc.append(acc_per_task_list)
                g_acc, g_auc = get_gacc(ratio=int(args.base_class / args.way), all_acc=all_acc)

                # Save model
                self.save_model(self.model, session, args.save_path)
                self.best_model_dict = deepcopy(self.model.state_dict())

                self.log_session_info(session, acc, g_acc, g_auc)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

        time_module.time_end()
        t_end_time_gpu = time_module.return_total_sec()

        t_end_time = time.time()
        total_time = t_end_time - t_start_time

        self.log_final_info(total_time, t_end_time_gpu)
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), self.result_list)

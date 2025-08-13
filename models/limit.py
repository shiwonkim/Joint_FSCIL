from models.base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader

from utils.helper import *
from utils.tools import *
from utils.metrics import *

from dataloader.data_utils import *
from dataloader.sampler import BasePreserverCategoriesSampler, NewCategoriesSampler
from models.archs.networks import LIMITNet


# copy from acastle.
class LIMIT(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path(epochs=self.args.pretrained_epochs)
        self.args = set_up_datasets(self.args)
        self.set_up_model()
        pass

    def set_up_model(self):
        self.model = LIMITNet(self.args, mode=self.args.base_mode)
        print(LIMITNet)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir != None:  #
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('*********WARNINGl: NO INIT MODEL**********')
            self.best_model_dict = None
            pass

    def update_param(self, model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def get_dataloader(self, session):
        if session == 0:
            trainset, train_fsl_loader, train_gfsl_loader, valloader, testloader = get_base_dataloader(self.args)
            return trainset, train_fsl_loader, train_gfsl_loader, valloader, testloader
        else:
            trainset, trainloader, testloader, train_fsl_loader = get_new_dataloader(self.args, session)
            return trainset, trainloader, testloader, train_fsl_loader

    def get_session_classes(self, session):
        class_list = np.arange(self.args.base_class + session * self.args.way)
        return class_list

    def get_optimizer_base(self):

        top_para = [v for k, v in self.model.named_parameters() if ('encoder' not in k and 'cls' not in k)]
        optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base},
                                     {'params': top_para, 'lr': self.args.lrg}],
                                    momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def train(self):
        args = self.args
        time_module = TimeCalculator()
        time_module.time_start()
        t_start_time = time.time()
        all_acc = []

        # init train statistics
        self.result_list = calc_gflops(self.result_list, self.model, (3, 32, 32))

        for session in range(args.start_session, args.sessions):
            if session == 0:
                train_set, train_fsl_loader, train_gfsl_loader, valloader, testloader = self.get_dataloader(session)
            else:
                train_set, trainloader, testloader, train_fsl_loader = self.get_dataloader(session)

            if self.best_model_dict is not None:
                self.model = self.update_param(self.model, self.best_model_dict)

            if session == 0:  # load base class train img label
                self.result_list.append('\nBase session pre-training...')
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    self.model.eval()
                    tl, ta = self.base_train(self.model, train_fsl_loader, train_gfsl_loader,
                                             optimizer, scheduler, epoch, args)
                    vl, va = validate(self.model, valloader, epoch, args, session, limit=True)

                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    self.model.module.mode = 'avg_cos'

                    # Validation
                    if (va * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (va * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        # self.save_model(self.model, session, args.save_path)
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

                # always replace fc with avg mean
                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    self.save_model(self.model, session, args.save_path)

                    self.model.module.mode = 'avg_cos'

                acc, _, acc_per_task_list = test(self.model, testloader, args, session, limit=True)
                all_acc.append(acc_per_task_list)
                g_acc, g_auc = get_gacc(ratio=int(args.base_class / args.way), all_acc=all_acc)

                self.log_session_info(session, acc, g_acc, g_auc)

            else:  # incremental learning sessions
                print("training session: [%d]" % session)
                self.model.load_state_dict(self.best_model_dict)
                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                train_fsl_loader.dataset.transform = testloader.dataset.transform

                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)
                acc, _, acc_per_task_list = test(self.model, testloader, args, session, limit=True)
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

    def base_train(self, model, train_fsl_loader, train_gfsl_loader, optimizer, scheduler, epoch, args):
        tl = Averager()
        ta = Averager()

        for _, batch in tqdm(enumerate(zip(train_fsl_loader, train_gfsl_loader))):
            support_data, support_label = batch[0][0].cuda(), batch[0][1].cuda()
            query_data, query_label = batch[1][0].cuda(), batch[1][1].cuda()
            model.module.mode = 'classifier'
            logits = model(support_data, query_data, support_label, epoch)
            logits = logits[:, :args.base_class]
            total_loss = F.cross_entropy(logits, query_label.view(-1, 1).repeat(1, args.num_tasks).view(-1))
            acc = count_acc(logits, query_label.view(-1, 1).repeat(1, args.num_tasks).view(-1))

            lrc = scheduler.get_last_lr()[0]
            # tqdm_gen.set_description('Session 0, epo {}, lrc={:.4f},total loss={:.4f} query acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_item = total_loss.item()

            del logits, total_loss
        print(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} query acc={:.4f}'.format(epoch, lrc, total_loss_item, acc))
        print('Self.current_way:', model.module.current_way)
        tl = tl.item()
        ta = ta.item()
        return tl, ta

from models.base import Trainer
from models.archs.networks import CECNet
from dataloader.data_utils import *

import torch.nn as nn
from copy import deepcopy

from utils.helper import *
from utils.tools import *
from utils.metrics import *


class CEC(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = CECNet(self.args, mode=self.args.base_mode)
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
    
    def get_base_optimizer_meta(self):
        optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base_meta},
                                     {'params': self.model.module.slf_attn.parameters(), 'lr': self.args.lrg}],
                                    momentum=self.args.momentum, nesterov=True, weight_decay=self.args.decay)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma_meta)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones, gamma=self.args.gamma_meta)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler
    
    def train(self):
        args = self.args
        time_module = TimeCalculator()
        time_module.time_start()
        t_start_time = time.time()
        all_acc = []

        self.result_list = calc_gflops(self.result_list, self.model, (3, 32, 32))

        for session in range(args.start_session, args.sessions):

            # Base session pre-training
            if session == 0:
                self.result_list.append('\nBase session pre-training...')

                self.model.load_state_dict(self.best_model_dict)
                train_set, trainloader, valloader, testloader = get_base_dataloader(self.args)

                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_base_optimizer(args, self.model)

                for epoch in range(args.epochs_base):
                    pre_start_time = time.time()

                    tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                    vl, va = validate(self.model, valloader, epoch, args, session)

                    # Save better model
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
                    
                    print('This epoch takes %d seconds' % (time.time() - pre_start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - pre_start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                self.model.load_state_dict(self.best_model_dict)
                if not args.not_data_init:
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    # self.save_model(self.model, session, args.save_path)
                    print('Replace the fc with average embedding...')
                    self.best_model_dict = deepcopy(self.model.state_dict())

                    self.model.module.mode = 'avg_cos'
            
            self.model = self.update_param(self.model, self.best_model_dict)

            # Pseudo incremental learning
            if session == 0:
                self.result_list.append('\nPseudo incremental learning...')

                train_set, trainloader, valloader, testloader = get_base_dataloader_meta(self.args)
                optimizer, scheduler = self.get_base_optimizer_meta()

                for epoch in range(args.epochs_base):
                    pseudo_start_time = time.time()
                    self.model.eval()

                    tl, ta = self.meta_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)

                    self.model.module.mode = 'avg_cos'

                    vl, va = self.meta_validate(self.model, valloader, epoch, args, session)

                    # Save better model
                    if (va * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (va * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        self.save_model(self.model, session, args.save_path)
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                    print('best epoch {}, best val acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                      self.trlog['max_acc'][session]))
                    
                    lrc = scheduler.get_last_lr()[0]
                    self.log_train_info(tl, ta, epoch, lrc, vl, va)

                    print('This epoch takes %d seconds' % (time.time() - pseudo_start_time),
                          '\nstill need around %.2f mins to finish' % (
                                  (time.time() - pseudo_start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                # Always replace fc with avg mean
                self.model.load_state_dict(self.best_model_dict)
                self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                self.save_model(self.model, session, args.save_path)
                print('Replace the fc with average embedding...')
                self.best_model_dict = deepcopy(self.model.state_dict())

                self.model.module.mode = 'avg_cos'
                
                acc, _, acc_per_task_list = self.meta_test(self.model, testloader, args, session)
                all_acc.append(acc_per_task_list)
                g_acc, g_auc = get_gacc(ratio=int(args.base_class / args.way), all_acc=all_acc)
                
                self.log_session_info(session, acc, g_acc, g_auc)

            # Incremental learning sessions
            else:
                print("training session: [%d]" % session)
                train_set, trainloader, testloader = get_new_dataloader(self.args, session)
                
                self.model.load_state_dict(self.best_model_dict)
                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                acc, _, acc_per_task_list = self.meta_test(self.model, testloader, args, session)
                all_acc.append(acc_per_task_list)
                g_acc, g_auc = get_gacc(ratio=int(args.base_class / args.way), all_acc=all_acc)

                # Save better model
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

    def update_param(self, model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        return model

    def replace_to_rotate(self, proto_tmp, query_tmp):
        for i in range(self.args.low_way):
            rot_list = [90, 180, 270]
            sel_rot = random.choice(rot_list)

            if sel_rot == 90:
                proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].transpose(2, 3).flip(2)
                query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].transpose(2, 3).flip(2)
            elif sel_rot == 180:
                proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].flip(2).flip(3)
                query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].flip(2).flip(3)
            elif sel_rot == 270:
                proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].transpose(2, 3).flip(3)
                query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].transpose(2, 3).flip(3)
        
        return proto_tmp, query_tmp
    
    def meta_train(self, model, trainloader, optimizer, scheduler, epoch, args):
        tl = Averager()
        ta = Averager()

        tqdm_gen = tqdm(trainloader)

        label = torch.arange(args.episode_way + args.low_way).repeat(args.episode_query)
        label = label.type(torch.cuda.LongTensor)

        for i, batch in enumerate(tqdm_gen, 1):
            data, true_label = [_.cuda() for _ in batch]

            k = args.episode_way * args.episode_shot
            proto, query = data[:k], data[k:]
            
            # Sample low_way data
            proto_tmp = deepcopy(
                proto.reshape(args.episode_shot, args.episode_way, proto.shape[1], proto.shape[2], proto.shape[3])[
                :args.low_shot,
                :args.low_way, :, :, :].flatten(0, 1))
            query_tmp = deepcopy(
                query.reshape(args.episode_query, args.episode_way, query.shape[1], query.shape[2], query.shape[3])[:,
                :args.low_way, :, :, :].flatten(0, 1))
            
            # Random choose rotate degree
            proto_tmp, query_tmp = self.replace_to_rotate(proto_tmp, query_tmp)

            model.module.mode = 'encoder'
            data = model(data)
            proto_tmp = model(proto_tmp)
            query_tmp = model(query_tmp)

            proto, query = data[:k], data[k:]

            proto = proto.view(args.episode_shot, args.episode_way, proto.shape[-1])
            query = query.view(args.episode_query, args.episode_way, query.shape[-1])

            proto_tmp = proto_tmp.view(args.low_shot, args.low_way, proto.shape[-1])
            query_tmp = query_tmp.view(args.episode_query, args.low_way, query.shape[-1])

            proto = proto.mean(0).unsqueeze(0)
            proto_tmp = proto_tmp.mean(0).unsqueeze(0)

            proto = torch.cat([proto, proto_tmp], dim=1)
            query = torch.cat([query, query_tmp], dim=1)

            proto = proto.unsqueeze(0)
            query = query.unsqueeze(0)

            logits = model.module._forward(proto, query)

            total_loss = F.cross_entropy(logits, label)

            acc = count_acc(logits, label)

            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                'Session 0, epo {}, lrc={:.4f}, total loss={:.4f}, acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        tl = tl.item()
        ta = ta.item()
        
        return tl, ta

    def meta_validate(self, model, dataloader, epoch, args, session):
        val_class = args.base_class + session * args.way
        model.eval()
        
        vl = Averager()
        va = Averager()
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader, 1):
                data, test_label = [_.cuda() for _ in batch]

                model.module.mode = 'encoder'
                query = model(data)
                query = query.unsqueeze(0).unsqueeze(0)

                proto = model.module.fc.weight[:val_class, :].detach()
                proto = proto.unsqueeze(0).unsqueeze(0)

                logits = model.module._forward(proto, query)
                loss = F.cross_entropy(logits, test_label)
                
                acc = count_acc(logits, test_label)

                vl.add(loss.item())
                va.add(acc)
            
            vl = vl.item()
            va = va.item()
            
            print('Epoch {}, val_loss={:.4f}, val_acc={:.4f}'.format(epoch, vl, va))
            
        return vl, va
    
    def meta_test(self, model, testloader, args, session, conf_matrix=False):
        test_class = args.base_class + session * args.way
        model = model.eval()
        
        acc_dic = {}
        ta = Averager()
        ta5 = Averager()

        lgt = torch.tensor([])
        lbs = torch.tensor([])
        
        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]

                model.module.mode = 'encoder'
                query = model(data)
                query = query.unsqueeze(0).unsqueeze(0)

                proto = model.module.fc.weight[:test_class, :].detach()
                proto = proto.unsqueeze(0).unsqueeze(0)

                logits = model.module._forward(proto, query)
                acc = count_acc(logits, test_label)
                top5_acc = count_acc_topk(logits, test_label)
                
                ta.add(acc)
                ta5.add(top5_acc)

                lgt = torch.cat([lgt, logits.cpu()])
                lbs = torch.cat([lbs, test_label.cpu()])

            ta = ta.item()
            ta5 = ta5.item()
            
            lgt = lgt.view(-1, test_class)
            lbs = lbs.view(-1)    

            acc_per_task, acc_per_task_list = count_acc_per_task(
                lgt, lbs.numpy(), init_task_size=args.base_class, task_size=args.way)

            if conf_matrix:
                save_model_dir = os.path.join(args.save_path, 'cm')
                ensure_path(save_model_dir)
                save_model_dir = os.path.join(
                    save_model_dir, 'session' + str(session) + '_cm')
                save_conf_matrix(lgt, lbs, save_model_dir)
            
            pred = torch.argmax(lgt, dim=1)
            cm = confusion_matrix(lbs, pred, normalize='true')
            per_cls_acc = cm.diagonal()
            seen_acc = np.mean(per_cls_acc[:args.base_class])
            unseen_acc = np.mean(per_cls_acc[args.base_class:])

            acc_dic['total'] = ta
            acc_dic['top5'] = ta5
            acc_dic['seen'] = seen_acc
            acc_dic['unseen'] = unseen_acc
        
        return acc_dic, acc_per_task, acc_per_task_list

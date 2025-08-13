from models.base import Trainer
from models.archs.networks import FACTNet
from dataloader.data_utils import *

import torch.nn as nn
from copy import deepcopy

from utils.helper import *
from utils.tools import *
from utils.metrics import *


class FACT(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = FACTNet(self.args, mode=self.args.base_mode)
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

        # Generate mask for dummy classifiers
        masknum = 3
        mask = np.zeros((args.base_class, args.num_classes))
        for i in range(args.num_classes - args.base_class):
            picked_dummy = np.random.choice(args.base_class, masknum, replace=False)
            mask[:, i+args.base_class][picked_dummy] = 1
        mask = torch.tensor(mask).cuda()

        for session in range(args.start_session, args.sessions):
            self.model.load_state_dict(self.best_model_dict)
            
            if session == 0:  # load base class train img label
                train_set, trainloader, valloader, testloader = get_base_dataloader(self.args)

                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_base_optimizer(args, self.model)

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    tl, ta = self.base_train(self.model, trainloader, optimizer, scheduler, epoch, args, mask)
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

                # Save dummy classifiers
                self.dummy_classifiers = deepcopy(self.model.module.fc.weight.detach())
                self.dummy_classifiers = F.normalize(self.dummy_classifiers[self.args.base_class:,:],p=2,dim=-1)
                self.old_classifiers = self.dummy_classifiers[:self.args.base_class,:]

            else:  # incremental learning sessions
                print("training session: [%d]" % session)
                train_set, trainloader, testloader = get_new_dataloader(self.args, session)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                acc, _, acc_per_task_list = self.test_intergrate(self.model, testloader, args, session)
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

    def base_train(self, model, trainloader, optimizer, scheduler, epoch, args, mask):
        tl = Averager()
        ta = Averager()
        model.train()

        tqdm_gen = tqdm(trainloader)

        for i, batch in enumerate(tqdm_gen, 1):
            beta = torch.distributions.beta.Beta(args.alpha, args.alpha).sample([]).item()
            data, train_label = [_.cuda() for _ in batch]

            logits = model(data)
            logits_ = logits[:, :args.base_class]
            loss = F.cross_entropy(logits_, train_label)
            acc = count_acc(logits_, train_label)
            
            if epoch >= args.loss_iter:
                logits_masked = logits.masked_fill(F.one_hot(train_label, num_classes=model.module.pre_allocate) == 1, -1e9)
                logits_masked_chosen = logits_masked * mask[train_label]
                pseudo_label = torch.argmax(logits_masked_chosen[:, args.base_class:], dim=-1) + args.base_class
                loss2 = F.cross_entropy(logits_masked, pseudo_label)

                index = torch.randperm(data.size(0)).cuda()
                pre_emb1 = model.module.pre_encode(data)
                mixed_data = beta * pre_emb1 + (1 - beta) * pre_emb1[index]
                mixed_logits = model.module.post_encode(mixed_data)

                newys = train_label[index]
                idx_chosen = newys!=train_label
                mixed_logits = mixed_logits[idx_chosen]

                pseudo_label1 = torch.argmax(mixed_logits[:, args.base_class:], dim=-1) + args.base_class # new class label
                pseudo_label2 = torch.argmax(mixed_logits[:, :args.base_class], dim=-1) # old class label
                loss3 = F.cross_entropy(mixed_logits, pseudo_label1)
                
                novel_logits_masked = mixed_logits.masked_fill(F.one_hot(pseudo_label1, num_classes=model.module.pre_allocate) == 1, -1e9)
                loss4 = F.cross_entropy(novel_logits_masked, pseudo_label2)
                
                total_loss = loss + args.balance * (loss2 + loss3 + loss4)
            else:
                total_loss = loss

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

    def test_intergrate(self, model, testloader, args, session, conf_matrix=False):
        test_class = args.base_class + session * args.way
        model = model.eval()

        acc_dic = {}
        ta = Averager()
        ta5 = Averager()

        lgt = torch.tensor([])
        lbs = torch.tensor([])

        proj_matrix = torch.mm(
            self.dummy_classifiers,
            F.normalize(torch.transpose(model.module.fc.weight[:test_class, :], 1, 0), p=2, dim=-1)
        )
        eta=args.eta

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                emb = model.module.encode(data)
            
                proj = torch.mm(
                    F.normalize(emb, p=2, dim=-1),
                    torch.transpose(self.dummy_classifiers, 1, 0)
                )
                topk, indices = torch.topk(proj, 40)
                res = (torch.zeros_like(proj))
                res_logit = res.scatter(1, indices, topk)

                logits1 = torch.mm(res_logit, proj_matrix)
                logits2 = model.module.forpass_fc(data)[:, :test_class] 
                logits = eta * F.softmax(logits1, dim=1) + (1-eta) * F.softmax(logits2, dim=1)
                
                acc = count_acc(logits, test_label)
                top5_acc = count_acc_topk(logits, test_label)
                
                ta.add(acc)
                ta5.add(top5_acc)
                
                lgt = torch.cat([lgt,logits.cpu()])
                lbs = torch.cat([lbs,test_label.cpu()])
            
            ta = ta.item()
            ta5= ta5.item()

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

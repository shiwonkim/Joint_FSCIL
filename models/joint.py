from models.base import Trainer
from models.archs.networks import JointNet
from dataloader.data_utils import *

import torch.nn as nn
from copy import deepcopy

from utils.helper import *
from utils.tools import *
from utils.metrics import *

from utils.losses import *
from utils.etf_head import *


class JointTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path(epochs=args.epochs_new)
        self.args = set_up_datasets(self.args)

        use_norm = True if args.ldam else False
        self.model = JointNet(self.args, mode=self.args.base_mode, use_norm=use_norm)
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
            self.last_model_dict = deepcopy(self.model.state_dict())
            self.init_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer(self, lr, epochs):
        # Set optimizer
        if self.args.etf:
            for p in self.model.module.fc.parameters():
                p.requires_grad = False
            params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            params = self.model.parameters()
        
        if self.args.optim == 'SGD':
            optimizer = torch.optim.SGD(params, lr, momentum=0.9, nesterov=True, weight_decay=self.args.decay)
        elif self.args.optim == 'Adam':
            optimizer = torch.optim.Adam(params, lr, weight_decay=self.args.decay)

        # Set scheduler
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones, gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        if self.args.imbsam:
            optimizer = ImbSAM(optimizer=optimizer, model=self.model)

        return optimizer, scheduler

    def train(self):
        args = self.args
        time_module = TimeCalculator()
        time_module.time_start()
        t_start_time = time.time()
        all_acc = []
        self.result_list = calc_gflops(self.result_list, self.model, (3, 32, 32))

        for session in range(args.start_session, args.sessions):
            self.model.load_state_dict(self.init_model_dict)

            if session == 0:  # load base class train img label
                train_set, trainloader, valloader, testloader = get_base_dataloader(self.args)

                if not args.no_training:
                    print('new classes for this session:\n', np.unique(train_set.targets))
                    optimizer, scheduler = self.get_optimizer(self.args.lr_new, self.args.epochs_base)

                    if self.args.etf:
                        criterion = 'reg_dot_loss'
                    else:
                        criterion = nn.CrossEntropyLoss()

                    for epoch in range(args.epochs_base):
                        start_time = time.time()

                        if args.imbsam:
                            tl, ta = base_train(self.model, trainloader, optimizer.optimizer, scheduler, epoch, args)
                        else:
                            tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                        # tl, ta = self.joint_train(self.model, trainloader, optimizer, scheduler,
                        #                           epoch, args, session, criterion)
                        vl, va = self.validate(self.model, valloader, epoch, args, session, criterion)

                        # Save better model
                        if (va * 100) >= self.trlog['max_acc'][session]:
                            self.trlog['max_acc'][session] = float('%.3f' % (va * 100))
                            self.trlog['max_acc_epoch'] = epoch
                            self.save_model(self.model, session, args.save_path, mode='joint')
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

                    # Evaluate the best model on the test set
                    self.model.load_state_dict(self.best_model_dict)
                else:
                    filename = '_best_acc.pth'
                    model_dir = os.path.join(args.save_path, 'session' + str(session) + filename)
                    self.model.load_state_dict(torch.load(model_dir)['params'])
                
                acc, _, acc_per_task_list = self.test(self.model, testloader, args, session)
                all_acc.append(acc_per_task_list)
                g_acc, g_auc = get_gacc(ratio=int(args.base_class / args.way), all_acc=all_acc)
                
                self.log_session_info(session, acc, g_acc, g_auc)

            else: # incremental learning sessions
                print("training session: [%d]" % session)
                optimizer, scheduler = self.get_optimizer(self.args.lr_new, self.args.epochs_new)
                train_set, trainloader, testloader = get_joint_dataloader(self.args, session)

                if not args.no_training:
                    head_classes, tail_classes = np.unique(train_set.datasets[0].targets), \
                                                 np.unique(train_set.datasets[1].targets)
                    print('new classes for this session:\n', head_classes, tail_classes)
                    tail_classes = torch.tensor(tail_classes)

                    # For balanced softmax, balanced loss, and LDAM
                    class_num = np.array([float(len(train_set.datasets[0]) // args.base_class)] * args.base_class +
                                        [float(args.shot_num)] * (session * args.way))

                    # For balanced loss
                    class_weights = 1.0 / class_num
                    class_weights = class_weights / class_weights.sum()
                    class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()

                    if self.args.etf:
                        criterion = 'reg_dot_loss'
                        if self.args.balanced_softmax:
                            criterion_2 = BalancedSoftmaxLoss(cls_num_list=torch.tensor(class_num).cuda()).cuda()
                            criterion = [criterion, criterion_2]
                        elif self.args.balanced_loss:
                            criterion_2 = nn.CrossEntropyLoss(weight=class_weights)
                            criterion = [criterion, criterion_2]
                        elif self.args.ldam:
                            criterion = [criterion, None]

                    elif self.args.balanced_softmax:
                        criterion = BalancedSoftmaxLoss(cls_num_list=torch.tensor(class_num).cuda()).cuda()
                    elif self.args.balanced_loss:
                        criterion = nn.CrossEntropyLoss(weight=class_weights)
                    else:
                        criterion = nn.CrossEntropyLoss()

                    for epoch in range(args.epochs_new):
                        if self.args.ldam:
                            per_cls_weights = None
                            if self.args.ldam_drw:
                                betas = [0, 0.9999, 0.9999, 0.9999]
                                idx = epoch // 160
                                effective_num = 1.0 - np.power(betas[idx], class_num)
                                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(class_num)
                                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
                            criterion_2 = LDAMLoss(cls_num_list=class_num, max_m=0.5, s=30, weight=per_cls_weights).cuda()

                            if self.args.etf:
                                criterion[1] = criterion_2
                            else:
                                criterion = criterion_2

                        start_time = time.time()
                        tl, ta = self.joint_train(self.model, trainloader, optimizer, scheduler, epoch,
                                                  args, session, criterion, tail_classes=tail_classes)

                        lrc = scheduler.get_last_lr()[0]
                        self.log_train_info(tl, ta, epoch, lrc)

                        print('This epoch takes %d seconds' % (time.time() - start_time),
                              '\nstill need around %.2f mins to finish this session' % (
                                      (time.time() - start_time) * (args.epochs_new - epoch) / 60))
                        scheduler.step()

                    # Save the last model
                    self.save_model(self.model, session, args.save_path, mode='joint')
                    self.last_model_dict = deepcopy(self.model.state_dict())

                else:
                    filename = '_last_acc.pth'
                    model_dir = os.path.join(args.save_path, 'session' + str(session) + filename)
                    self.model.load_state_dict(torch.load(model_dir)['params'])

                # Evaluate the last model on the test set
                # self.model.load_state_dict(self.last_model_dict)
                acc, _, acc_per_task_list = self.test(self.model, testloader, args, session)
                all_acc.append(acc_per_task_list)
                g_acc, g_auc = get_gacc(ratio=int(args.base_class / args.way), all_acc=all_acc)
                
                self.log_session_info(session, acc, g_acc, g_auc)

        time_module.time_end()
        t_end_time_gpu = time_module.return_total_sec()
        
        t_end_time = time.time()
        total_time = t_end_time - t_start_time

        self.log_final_info(total_time, t_end_time_gpu)
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), self.result_list)

        print('All acc, Seen acc, Unseen acc, Generalized AUC')
        print(self.trlog['max_acc'][-1])
        print(self.trlog['seen_acc'][-1])
        print(self.trlog['unseen_acc'][-1])
        print(self.trlog['g_auc'][-1])

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def joint_train(self, model, trainloader, optimizer, scheduler,
                    epoch, args, session, criterion, tail_classes=None):
        train_class = args.base_class + session * args.way
        tl = Averager()
        ta = Averager()
        
        model.train()
        
        if args.etf:
            encoder = model.module.encoder
            classifier = model.module.fc
        
        cmo_flag = (
            args.cmo and session > 0 and
            args.cmo_start_data_aug < epoch < (args.epochs_new - args.cmo_end_data_aug)
        )

        gen_new_iter = None
        weighted_gen_iter = None

        if args.batch_prop and session > 0:
            # tqdm_gen = tqdm(trainloader[0])
            # gen_new_iter = iter(trainloader[1])
            tqdm_gen = tqdm(trainloader)
        elif args.cmo and session > 0:
            tqdm_gen = tqdm(trainloader['trainloader'])
            weighted_gen_iter = iter(trainloader['weighted_trainloader'])
        else:
            tqdm_gen = tqdm(trainloader)

        for i, batch in enumerate(tqdm_gen, 1):
            data, train_label = [_.cuda() for _ in batch]
            
            if args.batch_prop and session > 0:
                # if i > len(tqdm_gen)//2:
                #     break

                # new_data_list = []
                # new_train_label_list = []
                
                # for _ in range((len(data)+1)//len(trainloader[1].dataset) + 1):
                #     try:
                #         da, trla = next(gen_new_iter)
                #     except StopIteration:
                #         gen_new_iter = iter(trainloader[1])
                #         da, trla = next(gen_new_iter)
                
                #     new_data_list.append(da)
                #     new_train_label_list.append(trla)
                
                # new_data_list = torch.stack(new_data_list, dim=0).reshape(-1, *new_data_list[0].shape[1:])[:len(data)].cuda()
                # new_train_label_list = torch.stack(new_train_label_list, dim=0).reshape(-1, *new_train_label_list[0].shape[1:])[:len(data)].cuda()
                
                # data = torch.concat([data, new_data_list], dim=0)
                # train_label = torch.concat([train_label, new_train_label_list], dim=0)
                pass
            
            if cmo_flag:
                try:
                    data2, train_label2 = next(weighted_gen_iter)
                except:
                    weighted_gen_iter = iter(trainloader['weighted_trainloader'])
                    data2, train_label2 = next(weighted_gen_iter)

                data2, train_label2 = data2.cuda(), train_label2.cuda()

            # CMO
            r = np.random.rand(1)
            if cmo_flag and r < args.cmo_mixup_prob:
                lam = np.random.beta(args.cmo_beta, args.cmo_beta)
                
                if args.cmo_mixup: # Mixup
                    data = lam * data + (1-lam) * data2
                else: # CutMix
                    bbx1, bby1, bbx2, bby2 = self.rand_bbox(data.size(), lam)
                    data[:, :, bbx1:bbx2, bby1:bby2] = data2[:, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
            
            if args.etf:
                cur_M = get_etf_matrix(args, classifier, train_label, mode='train')
                feat = encoder(data)
                feat = classifier(feat)
                logits = torch.matmul(feat, cur_M)[:, :train_class]

                with torch.no_grad():
                    feat_no_grad = feat.detach()
                    H_length = torch.clamp(torch.sqrt(torch.sum(feat_no_grad ** 2, dim=1, keepdims=False)), 1e-8)
            else:
                logits = model(data)[:, :train_class]
            
            if args.imbsam:
                if args.etf:
                    if cmo_flag and r < args.cmo_mixup_prob:
                        loss = etf_loss(args, dot_loss, logits, feat, train_label, cur_M, criterion, H_length,
                                        cmo=True, train_label2=train_label2, lam=lam)
                    else:
                        loss = etf_loss(args, dot_loss, logits, feat, train_label, cur_M, criterion, H_length,
                                        cmo=False)

                    loss.backward()
                    optimizer.first_step()
                    logits = model(data)[:, :train_class]

                init_criterion = criterion[1] if isinstance(criterion, list) else criterion
                init_criterion = nn.CrossEntropyLoss() if isinstance(init_criterion, str) else init_criterion
                if cmo_flag and r < args.cmo_mixup_prob:
                    loss = imbsam_loss(model, init_criterion, optimizer, data,
                                       [train_label, train_label2], logits, tail_classes, train_class, lam=lam)
                else:
                    loss = imbsam_loss(model, init_criterion, optimizer, data,
                                       train_label, logits, tail_classes, train_class)
            else:
                if cmo_flag and r < args.cmo_mixup_prob:
                    if args.etf:
                        loss = etf_loss(args, dot_loss, logits, feat, train_label, cur_M, criterion, H_length,
                                        cmo=True, train_label2=train_label2, lam=lam)
                    else:
                        loss = criterion(logits, train_label) * lam + criterion(logits, train_label2) * (1. - lam)
                else:
                    if args.etf:
                        loss = etf_loss(args, dot_loss, logits, feat, train_label, cur_M, criterion, H_length,
                                        cmo=False)
                    else:
                        loss = criterion(logits, train_label)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            acc = count_acc(logits, train_label)
            total_loss = loss

            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                'Session {}, epo {}, lrc={:.4f}, total loss={:.4f}, acc={:.4f}'.format(session, epoch+1, lrc, total_loss.item(), acc))
            tl.add(total_loss.item())
            ta.add(acc)
        
        tl = tl.item()
        ta = ta.item()

        return tl, ta

    def validate(self, model, dataloader, epoch, args, session, criterion):
        val_class = args.base_class + session * args.way
        model.eval()

        if args.etf:
            encoder = model.module.encoder
            classifier = model.module.fc
        
        vl = Averager()
        va = Averager()
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                
                if args.etf:
                    cur_M = get_etf_matrix(args, classifier, mode='val')
                    feat = encoder(data)
                    feat = classifier(feat)
                    logits = torch.matmul(feat, cur_M)[:, :val_class]

                    with torch.no_grad():
                        feat_no_grad = feat.detach()
                        H_length = torch.clamp(torch.sqrt(torch.sum(feat_no_grad ** 2, dim=1, keepdims=False)), 1e-8)
                    
                    loss = dot_loss(feat, test_label, cur_M, criterion, H_length, reg_lam=args.reg_lam)
                else:
                    logits = model(data)[:, :val_class]
                    loss = F.cross_entropy(logits, test_label)
                
                acc = count_acc(logits, test_label)

                vl.add(loss.item())
                va.add(acc)
            
            vl = vl.item()
            va = va.item()
            
            print('Epoch {}, val_loss={:.4f}, val_acc={:.4f}'.format(epoch, vl, va))
            
        return vl, va

    def test(self, model, testloader, args, session, conf_matrix=False):
        test_class = args.base_class + session * args.way
        model.eval()
        
        if args.etf:
            encoder = model.module.encoder
            classifier = model.module.fc

        acc_dic = {}
        ta = Averager()
        ta5 = Averager()

        lgt = torch.tensor([])
        lbs = torch.tensor([])

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                
                if args.etf:
                    cur_M = get_etf_matrix(args, classifier, mode='val')
                    feat = encoder(data)
                    feat = classifier(feat)
                    logits = torch.matmul(feat, cur_M)[:, :test_class]
                else:
                    logits = model(data)[:, :test_class]
                
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

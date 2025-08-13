import torch.nn as nn

from models.base import Trainer
from copy import deepcopy
from models.archs.networks import SAVCNet

from dataloader.data_utils import *
from dataloader.savc.fantasy import *

from utils.helper import *
from utils.tools import *
from utils.metrics import *
from utils.losses import SupContrastive

import torch, gc


class SAVC(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        if args.dataset in ('cifar100', 'cub200'):
            self.transform, self.num_trans = rotation2()
        elif args.dataset == 'mini_imagenet':
            self.transform, self.num_trans = rot_color_perm12()

        self.model = SAVCNet(self.args, mode=self.args.base_mode, trans=self.num_trans)
        # self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()
        self.criterion = SupContrastive().cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
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
            self.model.load_state_dict(self.best_model_dict)

            if session == 0:  # load base class train img label
                train_set, trainloader, valloader, testloader = get_dataloader(args, session)

                if not args.no_training:
                    train_set.multi_train = True
                    print('new classes for this session:\n', np.unique(train_set.targets))
                    optimizer, scheduler = self.get_optimizer_base()

                    for epoch in range(args.epochs_base):
                        gc.collect()
                        torch.cuda.empty_cache()
                        start_time = time.time()
                        # train base sess
                        tl, tl_joint, tl_moco, tl_moco_global, tl_moco_small, ta = self.base_train(self.model, trainloader,
                                                                                                   self.criterion, optimizer,
                                                                                                   scheduler, epoch,
                                                                                                   self.transform, args)
                        # test model with all seen class
                        vl, va = validate(self.model, valloader, epoch, args, session,
                                          savc=True, transform=self.transform)

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

                    if not args.not_data_init:
                        self.model.load_state_dict(self.best_model_dict)
                        train_set.multi_train = False
                        self.model = self.replace_base_fc(train_set, testloader.dataset.transform,
                                                          self.transform, self.model, args)
                        best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        torch.save(dict(params=self.model.state_dict()), best_model_dir)
                else:
                    train_set.multi_train = True
                    filename = '_best_acc.pth'
                    model_dir = os.path.join(args.save_path, 'session' + str(session) + filename)
                    self.model.load_state_dict(torch.load(model_dir)['params'])
                    # train_set.multi_train = False

                self.model.mode = self.args.new_mode
                acc, _, acc_per_task_list = test(self.model, testloader, args, session,
                                                 savc=True, transform=self.transform)
                all_acc.append(acc_per_task_list)
                g_acc, g_auc = get_gacc(ratio=int(args.base_class / args.way), all_acc=all_acc)

                self.log_session_info(session, acc, g_acc, g_auc)
            else:  # incremental learning sessions
                train_set, trainloader, testloader = get_dataloader(args, session)
                print("training session: [%d]" % session)

                self.model.mode = self.args.new_mode
                self.model.eval()
                train_transform = trainloader.dataset.transform
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.update_fc(trainloader, np.unique(train_set.targets), session, transform=self.transform)
                if args.incft:
                    trainloader.dataset.transform = train_transform
                    train_set.multi_train = True
                    self.update_fc_ft(trainloader, self.transform, self.model, self.num_trans, session, args)

                acc, _, acc_per_task_list = test(self.model, testloader, args, session,
                                                 savc=True, transform=self.transform)
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

    def base_train(self, model, trainloader, criterion, optimizer, scheduler, epoch, transform, args):
        tl = Averager()
        tl_joint = Averager()
        tl_moco = Averager()
        tl_moco_global = Averager()
        tl_moco_small = Averager()
        ta = Averager()
        model = model.train()
        tqdm_gen = tqdm(trainloader)

        for i, batch in enumerate(tqdm_gen, 1):
            data, single_labels = [_ for _ in batch]
            single_labels = single_labels.cuda()
            b, c, h, w = data[1].shape
            if len(args.num_crops) > 1:
                data_small = data[args.num_crops[0] + 1].unsqueeze(1)
                for j in range(1, args.num_crops[1]):
                    data_small = torch.cat((data_small, data[j + args.num_crops[0] + 1].unsqueeze(1)), dim=1)
                data_small = data_small.view(-1, c, args.size_crops[1], args.size_crops[1])
            else:
                data_small = None

            data_classify = transform(data[0].cuda())
            data_query = transform(data[1].cuda())
            data_key = transform(data[2].cuda())
            data_small = transform(data_small.cuda())
            m = data_query.size()[0] // b
            joint_labels = torch.stack([single_labels * m + ii for ii in range(m)], 1).view(-1)

            joint_preds, output_global, output_small, target_global, target_small = model(im_cla=data_classify,
                                                                                          im_q=data_query,
                                                                                          im_k=data_key,
                                                                                          labels=joint_labels,
                                                                                          im_q_small=data_small)
            loss_moco_global = criterion(output_global, target_global)
            loss_moco_small = criterion(output_small, target_small)
            loss_moco = args.alpha * loss_moco_global + args.beta * loss_moco_small

            joint_preds = joint_preds[:, :args.base_class * m]
            joint_loss = F.cross_entropy(joint_preds, joint_labels)

            agg_preds = 0
            for i in range(m):
                agg_preds = agg_preds + joint_preds[i::m, i::m] / m

            loss = joint_loss + loss_moco
            total_loss = loss

            acc = count_acc(agg_preds, single_labels)

            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            tl.add(total_loss.item())
            tl_joint.add(joint_loss.item())
            tl_moco_global.add(loss_moco_global.item())
            tl_moco_small.add(loss_moco_small.item())
            tl_moco.add(loss_moco.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tl = tl.item()
        ta = ta.item()
        tl_joint = tl_joint.item()
        tl_moco = tl_moco.item()
        tl_moco_global = tl_moco_global.item()
        tl_moco_small = tl_moco_small.item()
        return tl, tl_joint, tl_moco, tl_moco_global, tl_moco_small, ta

    def replace_base_fc(self, trainset, test_transform, data_transform, model, args):
        # replace fc.weight with the embedding average of train data
        model = model.eval()

        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                                  num_workers=8, pin_memory=True, shuffle=False)
        trainloader.dataset.transform = test_transform
        embedding_list = []
        label_list = []
        # data_list=[]
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                data, label = [_.cuda() for _ in batch]
                b = data.size()[0]
                data = data_transform(data)
                m = data.size()[0] // b
                labels = torch.stack([label * m + ii for ii in range(m)], 1).view(-1)

                model.mode = 'encoder'
                embedding = model(data)

                embedding_list.append(embedding.cpu())
                label_list.append(labels.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        proto_list = []

        for class_index in range(args.base_class * m):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)

        proto_list = torch.stack(proto_list, dim=0)

        model.fc.weight.data[:args.base_class * m] = proto_list

        return model

    def update_fc_ft(self, trainloader, data_transform, model, m, session, args):
        # incremental finetuning
        old_class = args.base_class + args.way * (session - 1)
        new_class = args.base_class + args.way * session
        new_fc = nn.Parameter(
            torch.rand(args.way * m, model.num_features, device="cuda"),
            requires_grad=True)
        new_fc.data.copy_(model.fc.weight[old_class * m: new_class * m, :].data)

        if args.dataset == 'mini_imagenet':
            optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new},
                                         {'params': model.encoder_q.fc.parameters(), 'lr': 0.05 * args.lr_new},
                                         {'params': model.encoder_q.layer4.parameters(), 'lr': 0.001 * args.lr_new}, ],
                                        momentum=0.9, dampening=0.9, weight_decay=0)

        if args.dataset == 'cub200':
            optimizer = torch.optim.SGD([{'params': new_fc, 'lr': args.lr_new}],
                                        momentum=0.9, dampening=0.9, weight_decay=0)

        elif args.dataset == 'cifar100':
            optimizer = torch.optim.Adam([{'params': new_fc, 'lr': args.lr_new},
                                          {'params': model.encoder_q.fc.parameters(), 'lr': 0.01 * args.lr_new},
                                          {'params': model.encoder_q.layer3.parameters(), 'lr': 0.02 * args.lr_new}],
                                         weight_decay=0)

        with torch.enable_grad():
            for epoch in range(args.epochs_new):
                for batch in trainloader:
                    data, single_labels = [_ for _ in batch]
                    b, c, h, w = data[1].shape
                    origin = data[0].cuda()
                    data[1] = data[1].cuda()
                    data[2] = data[2].cuda()
                    single_labels = single_labels.cuda()
                    if len(args.num_crops) > 1:
                        data_small = data[args.num_crops[0] + 1].unsqueeze(1)
                        for j in range(1, args.num_crops[1]):
                            data_small = torch.cat((data_small, data[j + args.num_crops[0] + 1].unsqueeze(1)), dim=1)
                        data_small = data_small.view(-1, c, args.size_crops[1], args.size_crops[1]).cuda()
                    else:
                        data_small = None
                data_classify = data_transform(origin)
                data_query = data_transform(data[1])
                data_key = data_transform(data[2])
                data_small = data_transform(data_small)
                joint_labels = torch.stack([single_labels * m + ii for ii in range(m)], 1).view(-1)

                old_fc = model.fc.weight[:old_class * m, :].clone().detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                features, _ = model.encode_q(data_classify)
                features.detach()
                logits = model.get_logits(features, fc)
                joint_loss = F.cross_entropy(logits, joint_labels)
                _, output_global, output_small, target_global, target_small = model(im_cla=data_classify,
                                                                                    im_q=data_query, im_k=data_key,
                                                                                    labels=joint_labels,
                                                                                    im_q_small=data_small,
                                                                                    base_sess=False, last_epochs_new=(
                                epoch == args.epochs_new - 1))
                loss_moco_global = self.criterion(output_global, target_global)
                loss_moco_small = self.criterion(output_small, target_small)
                loss_moco = args.alpha * loss_moco_global + args.beta * loss_moco_small
                loss = joint_loss + loss_moco
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.fc.weight.data[old_class * m: new_class * m, :].copy_(new_fc.data)
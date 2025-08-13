from models.base import Trainer
from models.archs.networks import S3CNet
from dataloader.data_utils import *

import torch.nn as nn
from copy import deepcopy

from utils.helper import *
from utils.tools import *
from utils.metrics import *


class S3C(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = S3CNet(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        self.old_model = S3CNet(self.args, mode=self.args.base_mode)
        self.old_model = nn.DataParallel(self.old_model, list(range(self.args.num_gpu)))
        self.old_model = self.old_model.cuda()

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

            if session == 0:
                train_set, trainloader, valloader, testloader = get_base_dataloader(self.args)

                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_base_optimizer(args, self.model)
                
                for epoch in range(args.epochs_base):
                    start_time = time.time()

                    tl, ta = self.base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                    vl, va = self.validate(self.model, valloader, args, session)

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
                    self.model = self.replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    self.save_model(self.model, session, args.save_path)
                    print('Replace the fc with average embedding...')
                    self.best_model_dict = deepcopy(self.model.state_dict())

                    self.model.module.mode = 'avg_cos'
                
                acc, _, acc_per_task_list = self.test(self.model, testloader, args, session, agg=True)
                all_acc.append(acc_per_task_list)
                g_acc, g_auc = get_gacc(ratio=int(args.base_class / args.way), all_acc=all_acc)
                
                self.log_session_info(session, acc, g_acc, g_auc)

            else: # incremental learning sessions
                print("training session: [%d]" % session)
                train_set, trainloader, testloader = get_new_dataloader(self.args, session)

                prev_class = (args.base_class + (session-1) * args.way)
                curr_class = (args.base_class + session * args.way)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
               
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)
                
                optimizer = torch.optim.SGD(self.model.parameters(),lr=self.args.lr_new,
                                            momentum=0.9, dampening=0.9, weight_decay=0)
                
                self.finetune_fc(self.model, trainloader, optimizer, prev_class, curr_class, args.epochs_new, args)

                acc, _, acc_per_task_list = self.test(self.model, testloader, args, session, agg=True)
                all_acc.append(acc_per_task_list)
                g_acc, g_auc = get_gacc(ratio=int(args.base_class / args.way), all_acc=all_acc)

                # Save model
                self.save_model(self.model, session, args.save_path)
                self.best_model_dict = deepcopy(self.model.state_dict())

                self.log_session_info(session, acc, g_acc, g_auc)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))
            
            self.update_sigma_protos_feature_output(trainloader, train_set, testloader.dataset.transform, self.model, args, session)
            self.model.module.mode = self.args.new_mode

        time_module.time_end()
        t_end_time_gpu = time_module.return_total_sec()

        t_end_time = time.time()
        total_time = t_end_time - t_start_time

        self.log_final_info(total_time, t_end_time_gpu)
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), self.result_list)

    def base_train(self, model, trainloader, optimizer, scheduler, epoch, args):
        tl = Averager()
        ta = Averager()
        model.train()

        tqdm_gen = tqdm(trainloader)
        
        for i, batch in enumerate(tqdm_gen, 1):
            data, train_label = [_.cuda() for _ in batch]
            
            ########## Self supervision ##########
            data = torch.stack([torch.rot90(data, k, (2, 3)) for k in range(4)], 1)
            data = data.view(-1, 3, 32, 32)
            train_label = torch.stack([train_label * 4 + k for k in range(4)], 1).view(-1)

            logits, _, _ = model(data, stochastic=True)
            logits = logits[:, :args.base_class*4]
            loss = F.cross_entropy(logits, train_label)
            acc = count_acc(logits, train_label)

            total_loss = loss

            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                'Session 0, epo {}, lrc={:.4f}, total loss={:.4f}, acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        tl = tl.item()
        ta = ta.item()
        
        return tl, ta
    
    def finetune_fc(self, model, trainloader, optimizer, prev_class, curr_class, num_epochs, args):
        model.train()
        for parameter in model.module.parameters():
            parameter.requires_grad = False

        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        for parameter in model.module.fc.parameters():
            parameter.requires_grad = True
        
        with torch.enable_grad():
            for epoch in range(num_epochs):
                for batch in trainloader:
                    inputs, label = [_.cuda() for _ in batch]
                    inputs = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1)
                    inputs = inputs.view(-1, 3, 32, 32)
                    label = torch.stack([label * 4 + k for k in range(4)], 1).view(-1)
                    
                    logits, feature, attn_op = model(inputs, stochastic=False)
                    
                    protos = args.proto_list
                    indexes = torch.randperm(protos.shape[0])
                    protos = protos[indexes]
                    temp_protos = protos.cuda()
                    
                    num_protos = temp_protos.shape[0]
                    
                    label_proto = torch.arange(prev_class).cuda()
                    label_proto = label_proto[indexes] * 4

                    temp_protos = torch.cat((temp_protos,feature))
                    label_proto = torch.cat((label_proto,label))
                    logits_protos = model.module.fc(temp_protos, stochastic=True)
                    
                    loss_proto = nn.CrossEntropyLoss()(logits_protos[:num_protos,:curr_class*4], label_proto[:num_protos]) * args.lamda_proto 
                    loss_ce = nn.CrossEntropyLoss()(logits_protos[num_protos:, :curr_class*4], label_proto[num_protos:])
                    
                    loss = loss_proto + loss_ce
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print('Epoch {}, Loss_CE:{:.4f}, Loss proto:{:.4f}, Loss:{:.4f}'.format(epoch, loss_ce, loss_proto, loss))

    def replace_base_fc(self, trainset, transform, model, args):
        model.eval()

        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                                  num_workers=8, pin_memory=True, shuffle=False)
        trainloader.dataset.transform = transform
        
        embedding_list = []
        label_list = []

        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                data, label = [_.cuda() for _ in batch]
                model.module.mode = 'encoder'
                embedding, _ = model(data, stochastic=False)

                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        proto_list = []

        for class_index in range(args.base_class):
            data_index = (label_list == class_index).nonzero()
            embedding_this = embedding_list[data_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0)
            proto_list.append(embedding_this)

        proto_list = torch.stack(proto_list, dim=0)

        model.module.fc.weight.data[:args.base_class] = proto_list

        return model

    def validate(self, model, testloader, args, session):
        val_class = args.base_class + session * args.way
        model = model.eval()

        vl = Averager()
        va = Averager()

        with torch.no_grad():
            for i, batch in enumerate(testloader):
                data, test_label = [_.cuda() for _ in batch]
                data = torch.stack([torch.rot90(data, k, (2, 3)) for k in range(4)], 1)
                data = data.view(-1, 3, 32, 32)
                
                logits, features, _ = model(data, stochastic=False)
                logits_0 = logits[0::4, 0:val_class*4:4]
                logits_90 = logits[1::4, 1:val_class*4:4]
                logits_180 = logits[2::4, 2:val_class*4:4]
                logits_270 = logits[3::4, 3:val_class*4:4]
                logits_agg = (logits_0 + logits_90 + logits_180 + logits_270) / 4

                logits_original = logits[0::4, :val_class*4:4]
                loss = F.cross_entropy(logits_original, test_label)
                acc = count_acc(logits_original, test_label)
                acc_agg = count_acc(logits_agg, test_label)

                vl.add(loss.item())
                va.add(acc)
            
            vl = vl.item()
            va = va.item()
            
        return vl, va

    def test(self, model, testloader, args, session, agg=True, conf_matrix=False):
        test_class = args.base_class + session * args.way
        model = model.eval()

        acc_dic = {}
        ta = Averager()
        ta5 = Averager()

        lgt = torch.tensor([])
        lbs = torch.tensor([])

        with torch.no_grad():
            for i, batch in enumerate(testloader):
                data, test_label = [_.cuda() for _ in batch]
                data = torch.stack([torch.rot90(data, k, (2, 3)) for k in range(4)], 1)
                data = data.view(-1, 3, 32, 32)
                
                logits, features, _ = model(data, stochastic=False)
                logits_0 = logits[0::4, 0:test_class*4:4]
                logits_90 = logits[1::4, 1:test_class*4:4]
                logits_180 = logits[2::4, 2:test_class*4:4]
                logits_270 = logits[3::4, 3:test_class*4:4]
                logits_agg = (logits_0 + logits_90 + logits_180 + logits_270) / 4

                if agg:
                    logits = logits_agg
                else:
                    logits = logits[0::4, :test_class*4:4]

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

    def update_sigma_protos_feature_output(self, trainloader, trainset, transform, model, args, session):
        model.eval()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                                  num_workers=8, pin_memory=True, shuffle=False)
        trainloader.dataset.transform = transform
        
        embedding_list = []
        label_list = []

        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                data, label = [_.cuda() for _ in batch]
                _, embedding, _ = model(data, stochastic=False)

                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        proto_list = []
        radius = []
        
        if session == 0:    
            for class_index in range(args.base_class):
                data_index = (label_list == class_index).nonzero()
                embedding_this = embedding_list[data_index.squeeze(-1)]
                
                feature_class_wise = embedding_this.numpy()
                cov = np.cov(feature_class_wise.T)
                radius.append(np.trace(cov)/64)
                
                embedding_this = embedding_this.mean(0)
                proto_list.append(embedding_this)
            
            args.radius = np.sqrt(np.mean(radius)) 
            args.proto_list = torch.stack(proto_list, dim=0)
        
        else:
            for class_index in np.unique(trainset.targets):
                data_index = (label_list == class_index).nonzero()
                embedding_this = embedding_list[data_index.squeeze(-1)]
                
                feature_class_wise = embedding_this.numpy()
                cov = np.cov(feature_class_wise.T)
                radius.append(np.trace(cov)/64)
                
                embedding_this = embedding_this.mean(0)
                proto_list.append(embedding_this)
            
            args.proto_list = torch.cat((args.proto_list, torch.stack(proto_list, dim=0)), dim=0)

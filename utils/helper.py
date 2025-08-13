from .tools import *
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from typing import Dict
from utils.losses import RkdDistance, RKdAngle, pairwise_kl_div_v2, get_distance_matrix
from utils.feat_rect_logger import LOGGER
from collections import defaultdict
from torch.distributions import Distribution
from dataloader.sampler import PKsampler_incr
log = LOGGER.LOGGER


def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model.train()
    
    tqdm_gen = tqdm(trainloader)
    
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        logits = model(data)
        logits = logits[:, :args.base_class]
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


def replace_base_fc(trainset, transform, model, args):
    '''
    Replace fc.weight with the embedding average of train data
    '''
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
            embedding = model(data)

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


def validate(model, dataloader, epoch, args, session,
             savc=False, limit=False, transform=None):
    val_class = args.base_class + session * args.way
    model.eval()
    
    vl = Averager()
    va = Averager()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            if savc:
                b = data.size()[0]
                data = transform(data)
                m = data.size()[0] // b
                joint_preds = model(data)
                joint_preds = joint_preds[:, :val_class * m]
                logits = 0
                for j in range(m):
                    logits = logits + joint_preds[j::m, j::m] / m
            elif limit:
                model.module.mode = 'encoder'
                query = model(data)
                query = query.unsqueeze(0).unsqueeze(0)
                logits = model.module.forward_many(query)
                logits = logits[:, :val_class]
            else:
                logits = model(data)
                logits = logits[:, :val_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)

            vl.add(loss.item())
            va.add(acc)
        
        vl = vl.item()
        va = va.item()
        
        #print('Epoch {}, val_loss={:.4f}, val_acc={:.4f}'.format(epoch, vl, va))
        
    return vl, va


def test(model, testloader, args, session, conf_matrix=False,
         savc=False, limit=False, transform=None, is_feat_rect=False):
    test_class = args.base_class + session * args.way
    model.eval()

    if is_feat_rect:
        all_predicts = {f'layer{i}': [] for i in args.output_blocks}
        all_predicts.update({'final': []})
    else:
        #lgt = torch.tensor([])
        #lbs = torch.tensor([])
        all_predicts = torch.tensor([])

    all_labels = []
    acc_dic = {}
    ta = Averager()
    ta5 = Averager()

    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            if savc:
                b = data.size()[0]
                data = transform(data)
                m = data.size()[0] // b
                joint_preds = model(data)
                joint_preds = joint_preds[:, :test_class * m]
                logits = 0
                for j in range(m):
                    logits = logits + joint_preds[j::m, j::m] / m
                all_predicts = torch.cat([all_predicts, logits.cpu()])
            elif is_feat_rect:
                feat, all_feature = model.module.encoder(data, return_all=True)
                fc_weight = model.module.fc['final'].weight
                logits = torch.mm(F.normalize(feat, p=2, dim=-1),
                                  F.normalize(fc_weight, p=2, dim=-1).T)[:, :test_class]
                all_predicts['final'].append(logits.cpu())

                for ii in args.output_blocks:
                    fc_weight = model.module.fc[f'final'].weight
                    feat = all_feature[f'layer{ii}']
                    logits = torch.mm(F.normalize(feat, p=2, dim=-1),
                                      F.normalize(fc_weight, p=2, dim=-1).T)[:, :test_class]
                    all_predicts[f'layer{ii}'].append(logits.cpu())
            elif limit:
                model.module.mode = 'encoder'
                query = model(data)
                query = query.unsqueeze(0).unsqueeze(0)
                logits = model.module.forward_many(query)
                logits = logits[:, :test_class]
                all_predicts = torch.cat([all_predicts, logits.cpu()])
            else:
                logits = model(data)
                logits = logits[:, :test_class]
                all_predicts = torch.cat([all_predicts, logits.cpu()])
            all_labels.append(test_label.cpu())

            acc = count_acc(logits, test_label)
            top5_acc = count_acc_topk(logits, test_label)

            ta.add(acc)
            ta5.add(top5_acc)

        ta = ta.item()
        ta5 = ta5.item()

        #all_labels = all_labels.view(-1)
        all_labels = torch.cat(all_labels).numpy()
        
        if is_feat_rect is False:
            all_predicts = all_predicts.view(-1, test_class)

            acc_per_task, acc_per_task_list = count_acc_per_task(
                all_predicts, all_labels, init_task_size=args.base_class, task_size=args.way)
        else:
            for k, v in all_predicts.items():
                cur_all_predicts = torch.cat(v)
                acc_per_task, acc_per_task_list = count_acc_per_task(cur_all_predicts, all_labels, init_task_size=args.base_class,
                                                        task_size=args.way)
                log.info(f'{k}\n' + '\t' + str(acc_per_task))

        if conf_matrix:
            save_model_dir = os.path.join(args.save_path, 'cm')
            ensure_path(save_model_dir)
            save_model_dir = os.path.join(
                save_model_dir, 'session' + str(session) + '_cm')
            save_conf_matrix(all_predicts, all_labels, save_model_dir)

        if is_feat_rect:
            all_predicts = all_predicts['final']
            all_predicts = torch.cat(all_predicts)  # 리스트 내부 텐서를 합쳐서 하나의 Tensor로 변환

        pred = torch.argmax(all_predicts, dim=1)
        cm = confusion_matrix(all_labels, pred, normalize='true')
        per_cls_acc = cm.diagonal()
        seen_acc = np.mean(per_cls_acc[:args.base_class])
        unseen_acc = np.mean(per_cls_acc[args.base_class:])

        acc_dic['total'] = ta
        acc_dic['top5'] = ta5
        acc_dic['seen'] = seen_acc
        acc_dic['unseen'] = unseen_acc
    
    return acc_dic, acc_per_task, acc_per_task_list


def test_rectification(encoder: torch.nn.Module,
                       feature_rectification: torch.nn.ModuleDict,
                       fc: torch.nn.ModuleDict,
                       testloader: torch.utils.data.DataLoader,
                       epoch: int,
                       args,
                       session: int,
                       conf_matrix=False,
                       ):
    log.info(f'Testing @ Epoch {epoch}:')
    test_class = args.base_class + session * args.way
    encoder = encoder.eval()
    for i in args.output_blocks:
        feature_rectification[f'layer{i}'] = feature_rectification[f'layer{i}'].eval()

    all_predicts = defaultdict(list)
    all_labels = []
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            feat_strong, features_all = encoder(data, return_all=True)
            for ii in args.output_blocks:
                cur_feat_recti = feature_rectification[f'layer{ii}']
                cur_feat_inter = features_all[f'layer{ii}']
                rectified_strong_feature = cur_feat_recti(cur_feat_inter, feat_strong)
                fc_weight = fc['final'].weight
                logits = F.linear(F.normalize(rectified_strong_feature, p=2, dim=-1),
                                  F.normalize(fc_weight, p=2, dim=-1))  # cosine embed
                logits = logits[:, :test_class]
                all_predicts[f'layer{ii}'].append(logits.cpu())

            all_labels.append(test_label.cpu())

    all_labels = torch.cat(all_labels).numpy()
    mixed_logits = 0
    for ii in args.output_blocks:
        cur_all_predicts = torch.cat(all_predicts[f'layer{ii}'])
        mixed_logits += cur_all_predicts / len(args.output_blocks)
        all_acc_per_task, _ = count_acc_per_task(cur_all_predicts, all_labels, init_task_size=args.base_class,
                                                task_size=args.way)
        #log.info(f'layer{ii}\n' + '\t' + str(all_acc_per_task))
    all_acc_per_task, all_acc_per_task_list = count_acc_per_task(mixed_logits, all_labels,
                                                                init_task_size=args.base_class, task_size=args.way)

    ta=Averager()
    ta5=Averager()

    all_labels_tensor = torch.tensor(all_labels).to(mixed_logits.device)
    acc = count_acc(mixed_logits, all_labels_tensor)
    top5_acc = count_acc_topk(mixed_logits, all_labels_tensor)

    pred = torch.argmax(mixed_logits, dim=1)
    cm = confusion_matrix(all_labels_tensor, pred.numpy(), normalize='true')
    per_cls_acc = cm.diagonal()
    seen_acc = np.mean(per_cls_acc[:args.base_class])  # Base class accuracy
    unseen_acc = np.mean(per_cls_acc[args.base_class:])

    #log.info(f'Ensemble\n' + '\t' + str(all_acc_per_task))

    acc_dic = {
        'total': acc,
        'top5': top5_acc,
        'seen': seen_acc,
        'unseen': unseen_acc
    }
    if conf_matrix:
        save_model_dir = os.path.join(args.save_path, 'cm')
        ensure_path(save_model_dir)
        save_model_dir = os.path.join(save_model_dir, f'session{session}_cm')
        save_conf_matrix(mixed_logits, all_labels_tensor, save_model_dir)

    return acc_dic, all_acc_per_task, all_acc_per_task_list

def val_feature_rectification(
        model: torch.nn.Module,
        valloader: torch.utils.data.DataLoader,
        args,):
    model.eval()
    # feature_rectification = model.feature_rectification
    # encoder = model.encoder
    fc = model.fc


    rkd = RkdDistance()

    vl = Averager()
    va = Averager()

    with torch.no_grad():
        all_labels = []
        all_logits = defaultdict(list)

        for batch in valloader:
            data, val_label = [_.cuda() for _ in batch]
            batch_size = data.size(0) 
            strong_feat, all_features = model.encoder(data, return_all=True)

            loss = 0
            for i in args.output_blocks:
                cur_feat_recti = model.feature_rectification[f'layer{i}']
                intermediate_feat = all_features[f'layer{i}']
                rectified_strong_feature = cur_feat_recti(intermediate_feat, strong_feat)

                loss_cos = torch.nn.functional.cosine_embedding_loss(
                    rectified_strong_feature, strong_feat, target=torch.ones_like(val_label).cuda()) * 0.1
                loss_rkd = rkd(rectified_strong_feature, intermediate_feat) * 1
                loss_kl = F.kl_div(
                    F.log_softmax(torch.mm(F.normalize(rectified_strong_feature, p=2, dim=-1),
                                           F.normalize(model.fc['final'].weight, p=2, dim=-1).T) * 16, dim=-1),
                    F.softmax(torch.mm(F.normalize(intermediate_feat, p=2, dim=-1),
                                       F.normalize(model.fc['final'].weight, p=2, dim=-1).T) * 16, dim=-1),
                    reduction='batchmean') * 0.1

                loss += loss_rkd + loss_cos + loss_kl
                
                logits = torch.mm(F.normalize(rectified_strong_feature, p=2, dim=-1),
                                  F.normalize(fc['final'].weight, p=2, dim=-1).T) * 16
                logits = logits[:, :args.base_class]
                all_logits[f'layer{i}'].append(logits.cpu())

            vl.add(loss.item())

            all_labels.append(val_label.cpu())

        all_labels = torch.cat(all_labels).numpy()
        mixed_logits = 0

        for ii in args.output_blocks:
            cur_all_logits = torch.cat(all_logits[f'layer{ii}'])
            mixed_logits += cur_all_logits / len(args.output_blocks)

        acc = (mixed_logits.argmax(dim=-1).numpy() == all_labels).mean()
        va.add(acc * 100)


    return vl.item(), va.item()
    

def train_feature_rectification_base(
        model: torch.nn.Module, 
        encoder: torch.nn.Module,
        feature_rectification: torch.nn.ModuleDict,
        fc: torch.nn.ModuleDict,
        args,
        dataloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        hyper_params,
        validating_interval: int = 10,
        session: int = 0,
):
    assert session == 0
    log.info(f'start from training rectification')
    feature_rectification.requires_grad = True
    optimized_parameters = [{'params': feature_rectification.parameters()}, ]
    optimizer = torch.optim.SGD(optimized_parameters, lr=args.lr_fr, momentum=0.9, dampening=0.9,
                                weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones_fr,
                                                     gamma=args.gamma_fr)

    all_classes = args.base_class + args.way * session

    device = 'cuda'
    if args.feature_rectification_rkd.lower() == 'angle':
        rkd = RKdAngle()
    else:
        rkd = RkdDistance()

    kl_rkd = lambda x, y: F.smooth_l1_loss(pairwise_kl_div_v2(x, x), pairwise_kl_div_v2(y, y), reduction='sum')
    p_cos = hyper_params.get('cos', 0.1)
    p_rkd = hyper_params.get('rkd', 1)
    p_kl = hyper_params.get('kl', 0)
    p_rkd_inter = hyper_params.get('rkd_inter', 0)
    p_rkd_intra = hyper_params.get('rkd_intra', 0)
    rkd_split = hyper_params.get('rkd_split', 'intraI_interI')
    extra_rkd_split = hyper_params.get('extra_rkd_split', 'interL')
    p_rkd_intra_extra = hyper_params.get('rkd_intra_extra', 0)
    p_rkd_inter_extra = hyper_params.get('rkd_inter_extra', 0)
    epoch = 0

    best_val_acc = 0.0
    best_model_state = None
    with torch.enable_grad():
        for epoch in range(args.epochs_fr):
            losses = defaultdict(lambda: defaultdict(Averager))  # to make defaultDict happy
            # for it in range(args.iter_fr):
            for i in args.output_blocks:
                feature_rectification[f'layer{i}'] = feature_rectification[f'layer{i}'].train()
            encoder = encoder.eval()
            for idx, batch in enumerate(dataloader):
                data, label = [_.cuda() for _ in batch]
                # extract strong features
                with torch.no_grad():
                    strong_feat, all_features = encoder(data, return_all=True)

                loss = 0

                for i in args.output_blocks:
                    cur_feat_recti = feature_rectification[f'layer{i}']
                    intermediate_feat = all_features[f'layer{i}']
                    rectified_strong_feature = cur_feat_recti(intermediate_feat, strong_feat)

                    # cosine distance
                    loss_cos = torch.nn.functional.cosine_embedding_loss(rectified_strong_feature, strong_feat,
                                                                         target=torch.ones_like(label).cuda()) * p_cos
                    # we enforce RKD to find out.
                    loss_rkd = rkd(rectified_strong_feature, intermediate_feat) * p_rkd
                    fc_weight_final = fc['final'].weight
                    fc_weight = fc[f'layer{i}'].weight
                    logits1 = torch.mm(F.normalize(rectified_strong_feature, p=2, dim=-1),
                                       F.normalize(fc_weight_final, p=2, dim=-1).T) * 16  # temp=16
                    logits_inter = torch.mm(F.normalize(intermediate_feat, p=2, dim=-1),
                                            F.normalize(fc_weight_final, p=2, dim=-1).T) * 16
                    logits1 = logits1[:, :all_classes]
                    logits_inter = logits_inter[:, :all_classes]
                    # logits2 = torch.mm(F.normalize(strong_feat, p=2, dim=-1),
                    #                   F.normalize(fc_weight, p=2, dim=-1).T) * 16  # temp=16
                    # loss_kl = kl_rkd(F.softmax(logits1, dim=-1), F.softmax(logits_orignal, dim=-1)) * p_kl
                    loss_kl = F.kl_div(F.log_softmax(logits1, dim=-1), F.softmax(logits_inter, dim=-1),
                                       reduction='batchmean') * p_kl

                    # get mask
                    # mask = torch.tensor([], dtype=torch.bool).to(label.device)
                    mask = []
                    for gt in label:
                        cur_mask = label == gt
                        mask.append(cur_mask)
                    mask = torch.stack(mask)
                    class_distance_main = get_distance_matrix(rectified_strong_feature)
                    class_distance_strong = get_distance_matrix(strong_feat, no_grad=True)
                    class_distance_intermediate = get_distance_matrix(intermediate_feat, no_grad=True)
                    intra_class_main = class_distance_main[mask]
                    inter_class_main = class_distance_main[~mask]
                    intra_class_strong = class_distance_strong[mask]
                    intra_class_intermediate = class_distance_intermediate[mask]
                    inter_class_strong = class_distance_strong[~mask]
                    inter_class_intermediate = class_distance_intermediate[~mask]
                    if 'intraL' in rkd_split:
                        loss_rkd_intra = F.smooth_l1_loss(intra_class_main, intra_class_strong) * p_rkd_intra
                    elif 'intraI' in rkd_split:
                        loss_rkd_intra = F.smooth_l1_loss(intra_class_main, intra_class_intermediate) * p_rkd_intra
                    else:
                        assert False, f'[intra]Expected intraL or intraI, but got {rkd_split}'

                    if 'interL' in rkd_split:
                        loss_rkd_inter = F.smooth_l1_loss(inter_class_main, inter_class_strong) * p_rkd_inter
                    elif 'interI' in rkd_split:
                        loss_rkd_inter = F.smooth_l1_loss(inter_class_main, inter_class_intermediate) * p_rkd_inter
                    else:
                        assert False, f'[inter]Expected interL or interI, but got {rkd_split}'
                    if 'interL' in extra_rkd_split:
                        loss_rkd_inter_extra = F.smooth_l1_loss(inter_class_main,
                                                                inter_class_strong) * p_rkd_inter_extra
                    else:
                        loss_rkd_inter_extra = 0
                    if 'intraL' in extra_rkd_split:
                        loss_rkd_intra_extra = F.smooth_l1_loss(intra_class_main,
                                                                intra_class_strong) * p_rkd_intra_extra
                    else:
                        loss_rkd_intra_extra = 0
                    with torch.no_grad():
                        acc = (logits1.argmax(1) == label).sum() / len(label)
                    cur_loss = loss_rkd + loss_cos + loss_kl + loss_rkd_inter + loss_rkd_intra
                    losses[f'layer{i}']['rkd'].add(loss_rkd.item())
                    losses[f'layer{i}']['cos'].add(loss_cos.item())
                    losses[f'layer{i}']['total'].add(cur_loss.item())
                    losses[f'layer{i}']['loss_kl'].add(loss_kl.item())
                    losses[f'layer{i}']['rkd_inter'].add(loss_rkd_inter.item())
                    losses[f'layer{i}']['rkd_intra'].add(loss_rkd_intra.item())
                    if loss_rkd_inter_extra != 0:
                        losses[f'layer{i}']['rkd_inter_extra'].add(loss_rkd_inter_extra)
                    if loss_rkd_intra_extra != 0:
                        losses[f'layer{i}']['rkd_intra_extra'].add(loss_rkd_intra_extra)
                    losses[f'layer{i}']['acc'].add(acc.item())
                    loss += cur_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #log.info(hyper_params)
            #log.info(get_logger_str(losses, args.output_blocks, epoch))
            scheduler.step()

            vl, va = val_feature_rectification(model, valloader, args)
            #log.info(f"Validation - Epoch [{epoch + 1}] - Loss: {va:.4f} - Acc: {va:.2f}%")

            if va > best_val_acc:
                best_val_acc = va
                from copy import deepcopy
                best_model_state = deepcopy(model.state_dict())
                #log.info(f"🔹 New best model found at Epoch {epoch + 1} with Acc: {va:.2f}%")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        log.info("🔹 Loaded the best model based on validation accuracy")

        #     if epoch % testing_interval == testing_interval - 1:
        #         test_rectification(encoder, feature_rectification, fc, testloader, epoch, args, session)

        # if not epoch % testing_interval == testing_interval - 1:
        #     test_rectification(encoder, feature_rectification, fc, testloader, epoch, args, session)

def train_feature_rectification_distribution(
        encoder: torch.nn.Module,
        feature_rectification: torch.nn.ModuleDict,
        target_fc: torch.nn.ModuleDict,
        args,
        MVN_distributions: dict,
        testloader: torch.utils.data.DataLoader,
        hyper_params: Dict,
        testing_interval: int = 1,
        session: int = 1,
        dataloader: torch.utils.data.DataLoader = None,
):
    for i in args.output_blocks:
        feature_rectification[f'layer{i}'].requires_grad = True

    optimized_parameters = [{'params': feature_rectification.parameters()}, ]
    optimizer = torch.optim.SGD(optimized_parameters, lr=args.lr_fr, momentum=0.9, dampening=0.9,
                                weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones_fr,
                                                     gamma=args.gamma_fr)

    all_classes = args.base_class + args.way * session

    distributions = {i: {'strong': MVN_distributions[i]['strong_distribution']} for i in range(all_classes)}
    for i in range(all_classes):
        for ii in args.output_blocks:
            distributions[i].update({f'layer{ii}': MVN_distributions[i][f'layer{ii}_distribution']})

    distributions_dataset = TensorAugDataset4(distributions, args.output_blocks, [i for i in range(all_classes)], )

    generator = torch.Generator()
    generator.manual_seed(0)
    sampler = PKsampler_incr(distributions_dataset, p=args.p, k=args.k)
    distributions_loader = torch.utils.data.DataLoader(distributions_dataset,
                                                       batch_size=args.batch_size_fr, num_workers=8,
                                                       collate_fn=my_collect_fn4,
                                                       sampler=sampler)

    device = 'cuda'
    if args.feature_rectification_rkd.lower() == 'angle':
        rkd = RKdAngle()
    else:
        rkd = RkdDistance()
    kl_rkd = lambda x, y: F.smooth_l1_loss(pairwise_kl_div_v2(x, x), pairwise_kl_div_v2(y, y))
    if dataloader is not None:
        data_iter = iter(dataloader)
        batch, data_iter = get_data(dataloader, data_iter)
        data, label = [_.cuda() for _ in batch]
    with torch.enable_grad():
        # distributions_iter = iter(distributions_loader)
        epoch = 0
        for epoch in range(args.epochs_fr):
            losses = defaultdict(lambda: defaultdict(Averager))
            for idx, fake_feats_batch in enumerate(distributions_loader):
                # feature_mapping = feature_mapping.train()
                # encoder = encoder.train()
                encoder = encoder.eval()
                feature_rectification.train()

                # for fake features
                # fake_feats_batch, distributions_iter = get_data(distributions_loader, distributions_iter)
                fake_feats, fake_label = (fake_feats_batch['feat'], fake_feats_batch['targets'])

                fake_feats_strong = fake_feats['final'].to(device)
                for i in args.output_blocks:
                    fake_feats[f'layer{i}'] = fake_feats[f'layer{i}'].to(device)
                fake_label = fake_label.to(device)

                # -----------------------for real data, with limited quantity------------------------------------
                if dataloader is not None:
                    with torch.no_grad():
                        strong_feat, all_block_feats = encoder(data, return_all=True)
                    fake_feats_strong = torch.cat((fake_feats_strong, strong_feat), dim=0)
                    fake_label = torch.cat((fake_label, label))

                    # extract weak features
                    for i in args.output_blocks:
                        cur_weak_feat = all_block_feats[f'layer{i}'].to(device)
                        fake_feats[f'layer{i}'] = torch.cat((fake_feats[f'layer{i}'], cur_weak_feat))

                # ------------------------feature rectify--------------------------------
                loss = 0
                p_cos = hyper_params.get('cos', 0.1)
                p_rkd = hyper_params.get('rkd', 1)
                p_ce_novel = hyper_params.get('ce_novel', 1)
                p_ce_current = hyper_params.get('ce_current', 1)
                p_ce_global = hyper_params.get('ce_global', 1)
                p_kl = hyper_params.get('kl', 0)
                p_rkd_intra = hyper_params.get('rkd_intra', 0)
                p_rkd_inter = hyper_params.get('rkd_inter', 0)
                rkd_split = hyper_params.get('rkd_split', 'intraI_interI')
                extra_rkd_split = hyper_params.get('extra_rkd_split', 'interL')
                p_rkd_intra_extra = hyper_params.get('rkd_intra_extra', 0)
                p_rkd_inter_extra = hyper_params.get('rkd_inter_extra', 0)
                current_classes = args.base_class + args.way * (session - 1) if session > 0 else args.base_class
                novel_mask = fake_label >= args.base_class
                current_mask = fake_label >= current_classes

                for i in args.output_blocks:
                    cur_inter_feat = fake_feats[f'layer{i}']
                    cur_rectified_strong_feature = feature_rectification[f'layer{i}'](cur_inter_feat, fake_feats_strong)

                    # cosine distance
                    loss_cos = torch.nn.functional.cosine_embedding_loss(cur_rectified_strong_feature,
                                                                         fake_feats_strong,
                                                                         target=torch.ones_like(
                                                                             fake_label).cuda()) * p_cos
                    # loss_cos = torch.nn.functional.cosine_embedding_loss(cur_rectified_strong_feature,
                    #                                                      cur_inter_feat,
                    #                                                      target=torch.ones_like(
                    #                                                          fake_label).cuda()) * p_cos
                    # loss_cos = 0
                    # we enforce RKD to find out.
                    loss_rkd = rkd(cur_rectified_strong_feature, cur_inter_feat) * p_rkd
                    # loss_rkd = 0

                    # ce loss current
                    current_fake_label_strong = fake_label[current_mask]
                    novel_fake_label_strong = fake_label[novel_mask]

                    fc_weight = target_fc[f'layer{i}'].weight
                    strong_fc_weight = target_fc['final'].weight
                    logits = torch.mm(F.normalize(cur_rectified_strong_feature, p=2, dim=-1),
                                      F.normalize(strong_fc_weight, p=2, dim=-1).T) * 16  # temp=16

                    logits = logits[:, :all_classes]
                    logits_current = logits[current_mask]
                    logits_novel = logits[novel_mask]

                    logits_inter = torch.mm(F.normalize(cur_inter_feat, p=2, dim=-1),
                                            F.normalize(strong_fc_weight, p=2, dim=-1).T) * 16
                    logits_inter = logits_inter[:, :all_classes]
                    # logits2 = torch.mm(F.normalize(strong_feat, p=2, dim=-1),
                    #                   F.normalize(fc_weight, p=2, dim=-1).T) * 16  # temp=16
                    # loss_kl = kl_rkd(logits, logits_orignal) * p_kl
                    loss_kl = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(logits_inter, dim=-1),
                                       reduction='batchmean') * p_kl

                    loss_fc_current = F.cross_entropy(logits_current, current_fake_label_strong) * p_ce_current
                    loss_fc_novel = F.cross_entropy(logits_novel, novel_fake_label_strong) * p_ce_novel
                    loss_fc_global = F.cross_entropy(logits, fake_label) * p_ce_global

                    # get mask
                    mask = []
                    for gt in fake_label:
                        cur_mask = fake_label == gt
                        mask.append(cur_mask)
                    mask = torch.stack(mask)
                    class_distance_main = get_distance_matrix(cur_rectified_strong_feature)
                    class_distance_strong = get_distance_matrix(fake_feats_strong, no_grad=True)
                    class_distance_intermediate = get_distance_matrix(cur_inter_feat, no_grad=True)
                    intra_class_main = class_distance_main[mask]
                    inter_class_main = class_distance_main[~mask]
                    intra_class_strong = class_distance_strong[mask]
                    intra_class_intermediate = class_distance_intermediate[mask]
                    inter_class_strong = class_distance_strong[~mask]
                    inter_class_intermediate = class_distance_intermediate[~mask]
                    if 'intraL' in rkd_split:
                        loss_rkd_intra = F.smooth_l1_loss(intra_class_main, intra_class_strong) * p_rkd_intra
                    elif 'intraI' in rkd_split:
                        loss_rkd_intra = F.smooth_l1_loss(intra_class_main, intra_class_intermediate) * p_rkd_intra
                    else:
                        assert False, f'[intra]Expected intraL or intraI, but got {rkd_split}'

                    if 'interL' in rkd_split:
                        loss_rkd_inter = F.smooth_l1_loss(inter_class_main, inter_class_strong) * p_rkd_inter
                    elif 'interI' in rkd_split:
                        loss_rkd_inter = F.smooth_l1_loss(inter_class_main, inter_class_intermediate) * p_rkd_inter
                    else:
                        assert False, f'[inter]Expected interL or interI, but got {rkd_split}'
                    if 'interL' in extra_rkd_split:
                        loss_rkd_inter_extra = F.smooth_l1_loss(inter_class_main,
                                                                inter_class_strong) * p_rkd_inter_extra
                    else:
                        loss_rkd_inter_extra = 0
                    if 'intraL' in extra_rkd_split:
                        loss_rkd_intra_extra = F.smooth_l1_loss(intra_class_main,
                                                                intra_class_strong) * p_rkd_intra_extra
                    else:
                        loss_rkd_intra_extra = 0
                    cur_loss = loss_rkd + loss_cos + loss_fc_current + loss_fc_global + loss_fc_novel + loss_kl + \
                               loss_rkd_inter + loss_rkd_intra + \
                               loss_rkd_inter_extra + loss_rkd_intra_extra
                    losses[f'layer{i}']['rkd'].add(loss_rkd)
                    losses[f'layer{i}']['cos'].add(loss_cos)
                    losses[f'layer{i}']['ce_cur'].add(loss_fc_current)
                    losses[f'layer{i}']['ce_nov'].add(loss_fc_novel)
                    losses[f'layer{i}']['ce_glo'].add(loss_fc_global)
                    losses[f'layer{i}']['loss_kl'].add(loss_kl)
                    losses[f'layer{i}']['rkd_inter'].add(loss_rkd_inter)
                    if loss_rkd_inter_extra != 0:
                        losses[f'layer{i}']['rkd_inter_extra'].add(loss_rkd_inter_extra)
                    if loss_rkd_intra_extra != 0:
                        losses[f'layer{i}']['rkd_intra_extra'].add(loss_rkd_intra_extra)
                    losses[f'layer{i}']['rkd_intra'].add(loss_rkd_intra)
                    losses[f'layer{i}']['loss'].add(cur_loss)
                    loss += cur_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            #print(optimizer.param_groups[0]["lr"])

            #log.info(hyper_params)
            #log.info(get_logger_str(losses, args.output_blocks, epoch))


        # if not epoch % testing_interval == testing_interval - 1:
        #     test_rectification(encoder, feature_rectification, target_fc, testloader, epoch, args, session)


class TensorAugDataset4(torch.utils.data.Dataset):  # multi layer implementation

    def __init__(self, dist_feat: Dict[int, Dict[str, Distribution]], output_blocks, all_targets, n=1, ):

        self.dist_feat = dist_feat
        self.all_targets = all_targets
        self.n = n
        self.num_cls = len(self.all_targets)
        self.output_blocks = output_blocks

    def __len__(self):
        return 600 * self.num_cls

    def __getitem__(self, idx):
        idx = idx % len(self.all_targets)
        y = self.all_targets[idx]
        feat = {'final': torch.tensor([])}
        feat.update({f'layer{i}': torch.tensor([]) for i in self.output_blocks})
        # feat_mean = torch.tensor([])
        y_out = torch.tensor([y for _ in range(self.n)])
        for _ in range(self.n):
            feat['final'] = torch.cat((feat['final'], self.dist_feat[y]['strong'].sample().unsqueeze(0)), dim=0)
            # feat_mean = torch.cat((feat_mean, self.feat_mean[y.item()].unsqueeze(0)), dim=0)

        for i in self.output_blocks:
            y_out = torch.tensor([y for _ in range(self.n)])
            for _ in range(self.n):
                feat[f'layer{i}'] = torch.cat((feat[f'layer{i}'], self.dist_feat[y][f'layer{i}'].sample().unsqueeze(0)),
                                              dim=0)
        return {"targets": y_out, "feat": feat, }

    
def my_collect_fn4(batch):
    targets = []
    feat = {f'{i}': [] for i in batch[0]['feat'].keys()}
    feat.update({'final': []})
    for i in batch:
        targets.append(i['targets'])
        for ii in i['feat'].keys():
            feat[ii].append(i['feat'][ii])
    targets = torch.cat(targets)
    for ii in feat.keys():
        feat[ii] = torch.cat(feat[ii])
    return {"targets": targets, "feat": feat}


def get_data(loader, data_iter):
    try:
        data = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        data = next(data_iter)
    return data, data_iter
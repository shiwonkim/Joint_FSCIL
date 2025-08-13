import torch
import random
import numpy as np
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from PIL import Image, ImageOps, ImageFilter

from dataloader.savc.constrained_cropping import CustomMultiCropping
from dataloader.sampler import NewCategoriesSampler, CategoriesSampler, \
    PKsampler_base, get_weighted_sampler


def set_up_datasets(args):
    ''' CIFAR100 '''
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = args.shot_num
        args.sessions = 9
    
    ''' CUB200 '''
    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = args.shot_num
        args.sessions = 11
    
    ''' MiniImageNet '''
    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = args.shot_num
        args.sessions = 9
    
    ''' ImageNet '''
    if args.dataset == 'imagenet100':
        import dataloader.imagenet100.ImageNet as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = args.shot_num
        args.sessions = 9
    if args.dataset == 'imagenet1000':
        import dataloader.imagenet1000.ImageNet as Dataset
        args.base_class = 600
        args.num_classes = 1000
        args.way = 50
        args.shot = args.shot_num
        args.sessions = 9

    if args.end_session is not None:
        args.sessions = args.end_session + 1

    args.Dataset = Dataset
    return args


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class DataAugmentationINCRE:
    def __init__(self, data='CIFAR'):
        if data == 'CIFAR':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.8, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                        std=[0.2675, 0.2565, 0.2761]),
            ])
        elif data == 'CUB200':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomRotation(20),
                GaussianBlur(p=0.5, radius_min=0.1, radius_max=2.0),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        elif data == 'MINI':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(84, scale=(0.6, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomRotation(20),
                GaussianBlur(p=0.5, radius_min=0.1, radius_max=2.0),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])

    def __call__(self, image):
        return self.transform(image)


def train_val_split(trainset, val_ratio, seed):
    ''' Split the original training set into train and validation sets '''
    
    full_len = len(trainset)
    val_len = int(full_len * val_ratio)
    train_len = full_len - val_len

    gen = torch.Generator()
    gen.manual_seed(seed)

    train_set, val_set = torch.utils.data.random_split(trainset, [train_len, val_len], generator=gen)

    return train_set, val_set


def train_val_split_stratified(trainset, val_ratio, seed):
    ''' Split the original training set into train and validation sets (stratified) '''

    rng = np.random.default_rng(seed)
    
    # Group indices by class
    class_indices = defaultdict(list)
    for idx, target in enumerate(trainset.targets):
        class_indices[target].append(idx)

    train_indices, val_indices = [], []

    # Split indices for each class
    for _, indices in class_indices.items():
        indices = rng.permutation(indices) # shuffle indices
        split = int(len(indices) * val_ratio)
        
        val_indices.extend(indices[:split])
        train_indices.extend(indices[split:])
    
    train_set = torch.utils.data.Subset(trainset, train_indices)
    val_set = torch.utils.data.Subset(trainset, val_indices)

    return train_set, val_set


def get_dataloader(args, session):
    if session == 0:
        trainset, trainloader, valloader, testloader = get_base_dataloader(args)
        return trainset, trainloader, valloader, testloader
    else:
        trainset, trainloader, testloader = get_new_dataloader(args, session)
        return trainset, trainloader, testloader


def get_base_dataloader(args):
    class_index = np.arange(args.base_class)
    savc_transform = get_transform(args) if args.project == 'savc' else None
    return_limit_meta_loader = True if args.project == 'limit' else False

    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True,
                                         val_ratio=args.val_ratio, seed=args.seed,
                                         savc_transform=savc_transform,
                                         img_size=args.image_size, is_feat_rect=False)

        valset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=True,
                                       index=class_index, base_sess=True,
                                       val=True, val_ratio=args.val_ratio, seed=args.seed, img_size=args.image_size, is_feat_rect=False)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True, img_size=args.image_size, is_feat_rect=False)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True, index=class_index, base_sess=True,
                                       val_ratio=args.val_ratio, seed=args.seed,
                                       savc_transform=savc_transform,
                                       img_size=args.image_size, is_feat_rect=False)
        valset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index, base_sess=True,
                                     val=True, val_ratio=args.val_ratio, seed=args.seed, img_size=args.image_size, is_feat_rect=False)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index, img_size=args.image_size, is_feat_rect=False)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index=class_index, base_sess=True,
                                             val_ratio=args.val_ratio, seed=args.seed,
                                             savc_transform=savc_transform, img_size=args.image_size, is_feat_rect=False)
        valset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index, base_sess=True,
                                           val=True, val_ratio=args.val_ratio, seed=args.seed, img_size=args.image_size, is_feat_rect=False)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index, img_size=args.image_size, is_feat_rect=False)

    if args.project == 'feat_rect':
        sampler = PKsampler_base(trainset, p=args.p, k=args.k)
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=not args.no_pin_memory)
        trainloader_pk = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, sampler=sampler,
                                                num_workers=args.num_workers, pin_memory=not args.no_pin_memory)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base,
                                                  shuffle=True, num_workers=args.num_workers, pin_memory=not args.no_pin_memory)

    valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=len(valset) if len(valset) < args.test_batch_size else args.test_batch_size,
                                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size,
                                             shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # LIMIT Dataloader
    if return_limit_meta_loader:
        train_sampler = CategoriesSampler(trainset.targets, len(trainloader), args.sample_class,
                                          args.sample_shot)
        train_fsl_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                       batch_sampler=train_sampler,
                                                       num_workers=args.num_workers,
                                                       pin_memory=True)
        return trainset, train_fsl_loader, trainloader, valloader, testloader

    if args.project == 'feat_rect':
        return trainset, trainloader, valloader, testloader, trainloader_pk

    return trainset, trainloader, valloader, testloader


def get_base_dataloader_meta(args):
    txt_path = "data/index_list/" + args.dataset + "/session_1" + '.txt'
    class_index = np.arange(args.base_class)
    
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True,
                                         val_ratio=args.val_ratio, seed=args.seed)
        valset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=True,
                                       index=class_index, base_sess=True,
                                       val=True, val_ratio=args.val_ratio, seed=args.seed)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True, index_path=txt_path,
                                       val_ratio=args.val_ratio, seed=args.seed)
        valset = args.Dataset.CUB200(root=args.dataroot, train=False, index_path=txt_path,
                                     val=True, val_ratio=args.val_ratio, seed=args.seed)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)
    
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index_path=txt_path,
                                             val_ratio=args.val_ratio, seed=args.seed)
        valset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index_path=txt_path,
                                           val=True, val_ratio=args.val_ratio, seed=args.seed)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)

    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler,
                                              num_workers=args.num_workers, pin_memory=True)
    valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=len(valset),
                                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size,
                                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, valloader, testloader


def get_new_dataloader(args, session):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    savc_transform = get_transform(args) if args.project == 'savc' else None
    return_limit_meta_loader = True if args.project == 'limit' else False
    
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False, savc_transform=savc_transform, img_size=args.image_size, is_feat_rect=False) # 40 classes, 5 images per class (5 shot support set)
    
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True, index_path=txt_path,
                                       savc_transform=savc_transform, img_size=args.image_size, is_feat_rect=False)
    
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index_path=txt_path,
                                             savc_transform=savc_transform, img_size=args.image_size, is_feat_rect=False)
    
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = args.Dataset.ImageNet(root=args.dataroot, train=True, index_path=txt_path)

    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # Test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False, img_size=args.image_size, is_feat_rect=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_new, img_size=args.image_size, is_feat_rect=False)
    
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_new, img_size=args.image_size, is_feat_rect=False)
    
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        testset = args.Dataset.ImageNet(root=args.dataroot, train=False, index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    if return_limit_meta_loader:
        test_sampler = NewCategoriesSampler(trainset.targets, 1, 5, 5)
        train_fsl_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                       batch_sampler=test_sampler,
                                                       num_workers=0,
                                                       pin_memory=True)
        return trainset, trainloader, testloader, train_fsl_loader

    return trainset, trainloader, testloader


def get_transform(args):
    if args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
    if args.dataset == 'cub200':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    if args.dataset == 'mini_imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    assert (len(args.size_crops) == 2)
    crop_transform = CustomMultiCropping(size_large=args.size_crops[0],
                                         scale_large=(args.min_scale_crops[0], args.max_scale_crops[0]),
                                         size_small=args.size_crops[1],
                                         scale_small=(args.min_scale_crops[1], args.max_scale_crops[1]),
                                         N_large=args.num_crops[0], N_small=args.num_crops[1],
                                         condition_small_crops_on_key=args.constrained_cropping)

    if len(args.auto_augment) == 0:
        print('No auto augment - Apply regular moco v2 as secondary transform')
        secondary_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
#             transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            normalize])

    else:
        from dataloader.savc.auto_augment.auto_augment import AutoAugment
        from dataloader.savc.auto_augment.random_choice import RandomChoice
        print('Auto augment - Apply custom auto-augment strategy')
        counter = 0
        secondary_transform = []

        for i in range(len(args.size_crops)):
            for j in range(args.num_crops[i]):
                if not counter in set(args.auto_augment):
                    print('Crop {} - Apply regular secondary transform'.format(counter))
                    secondary_transform.extend([transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        normalize])])

                else:
                    print('Crop {} - Apply auto-augment/regular secondary transform'.format(counter))
                    trans1 = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        # AutoAugment(),
                        transforms.ToTensor(),
                        normalize])

                    trans2 = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                        normalize])

                    secondary_transform.extend([RandomChoice([trans1, trans2])])

                counter += 1
    return crop_transform, secondary_transform


def get_batch_prop_loader(args, trainset_base, trainset_new):
    trainloader_base = torch.utils.data.DataLoader(dataset=trainset_base, batch_size=args.batch_size_base//2, shuffle=True,
                                                   num_workers=args.num_workers, pin_memory=True)
    trainloader_new = torch.utils.data.DataLoader(dataset=trainset_new, batch_size=args.batch_size_base//2, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)
    
    return [trainloader_base, trainloader_new]


def get_joint_dataloader(args, session):
    txt_paths = ["data/index_list/" + args.dataset + "/session_" + str(i + 1) + '.txt' for i in range(session + 1)]
    
    base_class_txt = txt_paths[0]
    incr_class_txt = txt_paths[1:]

    base_class_idx = []
    with open(base_class_txt, 'r') as f:
        base_class_idx.extend(f.read().splitlines())

    incr_class_idx = []
    for txt_path in incr_class_txt:
        with open(txt_path, 'r') as f:
            incr_class_idx.extend(f.read().splitlines())

    if args.data_aug:
        transform_cifar = DataAugmentationINCRE(data='CIFAR')
        transform_cub = DataAugmentationINCRE(data='CUB200')
        transform_mini = DataAugmentationINCRE(data='MINI')
    else:
        transform_cifar, transform_cub, transform_mini = None, None, None

    if args.dataset == 'cifar100':
        base_trainset = args.Dataset.CIFAR100Joint(root=args.dataroot, train=True, download=False, index=base_class_idx)
        samples_per_class = int(len(base_trainset.data) / len(np.unique(base_trainset.targets)))
        incr_trainset = args.Dataset.CIFAR100Joint(root=args.dataroot, train=True, download=False,
                                                   index=incr_class_idx, transform=transform_cifar,
                                                   extra_data_path=args.deepsmote_path if args.deepsmote and args.project == 'joint' else None,
                                                   samples_per_class=samples_per_class if args.batch_prop_advanced else None)
    elif args.dataset == 'cub200':
        base_trainset = args.Dataset.CUB200Joint(root=args.dataroot, train=True, index_path=base_class_txt)
        incr_trainset = args.Dataset.CUB200Joint(root=args.dataroot, train=True,
                                                 index_path=incr_class_txt, transform=transform_cub)

    elif args.dataset == 'mini_imagenet':
        base_trainset = args.Dataset.MiniImageNetJoint(root=args.dataroot, train=True, index_path=base_class_txt)
        incr_trainset = args.Dataset.MiniImageNetJoint(root=args.dataroot, train=True,
                                                       index_path=incr_class_txt, transform=transform_mini)
        
    combined_trainset = ConcatDataset([base_trainset, incr_trainset])
    
    if args.batch_prop:
        #trainloader = get_batch_prop_loader(args, base_trainset, incr_trainset)
        cls_num_list = np.unique(base_trainset.targets, return_counts=True)[1].tolist() + \
                       np.unique(incr_trainset.targets, return_counts=True)[1].tolist()
        all_targets = base_trainset.targets.tolist() + incr_trainset.targets.tolist()
        weighted_sampler = get_weighted_sampler(args, cls_num_list, all_targets, cmo_flag=False)
        trainloader = torch.utils.data.DataLoader(dataset=combined_trainset, batch_size=args.batch_size_base,
                                                    num_workers=args.num_workers, pin_memory=True, sampler=weighted_sampler)
    elif args.cmo:
        cls_num_list = np.unique(base_trainset.targets, return_counts=True)[1].tolist() + \
                       np.unique(incr_trainset.targets, return_counts=True)[1].tolist()
        if isinstance(base_trainset.targets, list):
            all_targets = base_trainset.targets + incr_trainset.targets
        else:
            all_targets = base_trainset.targets.tolist() + incr_trainset.targets.tolist()
        weighted_sampler = get_weighted_sampler(args, cls_num_list, all_targets, cmo_flag=True)
        trainloader = torch.utils.data.DataLoader(dataset=combined_trainset, batch_size=args.batch_size_base,
                                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)
        weighted_trainloader = torch.utils.data.DataLoader(dataset=combined_trainset, batch_size=args.batch_size_base,
                                                           num_workers=args.num_workers, pin_memory=True, sampler=weighted_sampler)
        trainloader = {'trainloader': trainloader,
                       'weighted_trainloader': weighted_trainloader}
    else:
        trainloader = torch.utils.data.DataLoader(dataset=combined_trainset, batch_size=args.batch_size_base,
                                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # Test on all encountered classes
    class_new = get_session_classes(args, session)
    print("class_new", class_new)
    
    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100Joint(root=args.dataroot, train=False, download=False,
                                             index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200Joint(root=args.dataroot, train=False, index=class_new)
    
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNetJoint(root=args.dataroot, train=False, index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size,
                                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return combined_trainset, trainloader, testloader


def get_session_classes(args, session):
    class_list = np.arange(args.base_class + session * args.way)
    
    return class_list
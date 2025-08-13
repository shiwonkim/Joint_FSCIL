import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .autoaugment import AutoAugImageNetPolicy


class MiniImageNet(Dataset):

    def __init__(self, root='/data', train=True, transform=None, index_path=None,
                 index=None, base_sess=None, autoaug=0, val=None, val_ratio=0, seed=None,
                 savc_transform=None, img_size=84, is_feat_rect=False):
        if train or val:
            setname = 'train'
        else:
            setname = 'test'
        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        self.val = val
        self.val_ratio = val_ratio
        self.seed = seed

        self.IMAGE_PATH = os.path.join(root, 'miniimagenet/images')
        self.SPLIT_PATH = os.path.join(root, 'miniimagenet/split')

        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb

        
        if train:
            if transform is not None:
                self.transform = transform
            else:
                image_size = 84
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])
            if base_sess:
                # self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                self.data, self.targets = self.SelectfromClassesSplit(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxtSplit(self.data2label, index_path)
        else:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
            if self.val:
                if base_sess:
                    self.data, self.targets = self.SelectfromClassesSplit(self.data, self.targets, index)
                else:
                    self.data, self.targets = self.SelectfromTxtSplit(self.data2label, index_path)
            else:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
        
        if autoaug == 1:
            if train:
                image_size = 84
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    #add autoaug
                    AutoAugImageNetPolicy(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])
                if base_sess:
                    # self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                    self.data, self.targets = self.SelectfromClassesSplit(self.data, self.targets, index)
                else:
                    self.data, self.targets = self.SelectfromTxtSplit(self.data2label, index_path)
            else:
                image_size = 84
                self.transform = transforms.Compose([
                    transforms.Resize([92, 92]),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])
                if self.val:
                    if base_sess:
                        self.data, self.targets = self.SelectfromClassesSplit(self.data, self.targets, index)
                    else:
                        self.data, self.targets = self.SelectfromTxtSplit(self.data2label, index_path)
                else:
                    self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

        self.multi_train = False
        if savc_transform is not None:
            self.crop_transform = savc_transform[0]
            self.secondary_transform = savc_transform[1]
            if isinstance(self.secondary_transform, list):
                assert (len(self.secondary_transform) == self.crop_transform.N_large + self.crop_transform.N_small)

    def SelectfromTxt(self, data2label, index_path):
        #select from txt file, and make cooresponding mampping.
        index=[]
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        for line in lines:
            index.append(line.split('/')[3])
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.IMAGE_PATH, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromTxtSplit(self, data2label, index_path):
        index = []
        if isinstance(index_path, list):
            lines = []
            for ip in index_path:
                lines += [x.strip() for x in open(ip, 'r').readlines()]
        else:
            lines = [x.strip() for x in open(index_path, 'r').readlines()]

        for line in lines:
            index.append(line.split('/')[3])

        data_by_class = {}
        for i in index:
            img_path = os.path.join(self.IMAGE_PATH, i)
            label = data2label[img_path]
            if label not in data_by_class:
                data_by_class[label] = []
            data_by_class[label].append(img_path)
        
        data_tmp = []
        targets_tmp = []
        rng = np.random.default_rng(self.seed)

        for label, img_paths in data_by_class.items():
            img_paths = rng.permutation(img_paths)
            split = int(len(img_paths) * self.val_ratio)

            if self.val:
                selected_paths = img_paths[:split]
            else:
                selected_paths = img_paths[split:]

            data_tmp.extend(selected_paths)
            targets_tmp.extend([label] * len(selected_paths))

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        #select from csv file, choose all instances from this class.
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def SelectfromClassesSplit(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        rng = np.random.default_rng(self.seed)
        for i in index:
            ind_cl = np.where(i == targets)[0]
            ind_cl = rng.permutation(ind_cl)

            split = int(len(ind_cl) * self.val_ratio)
            if self.val:
                ind = ind_cl[:split]
            else:
                ind = ind_cl[split:]

            for j in ind:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        if self.multi_train:
            image = Image.open(path).convert('RGB')
            classify_image = [self.transform(image)]
            multi_crop, multi_crop_params = self.crop_transform(image)
            assert (len(multi_crop) == self.crop_transform.N_large + self.crop_transform.N_small)
            if isinstance(self.secondary_transform, list):
                multi_crop = [tf(x) for tf, x in zip(self.secondary_transform, multi_crop)]
            else:
                multi_crop = [self.secondary_transform(x) for x in multi_crop]
            image = classify_image + multi_crop
        else:
            image = self.transform(Image.open(path).convert('RGB'))
        return image, targets


class MiniImageNet_concate(Dataset):
    def __init__(self, train,x1,y1,x2,y2):
        
        if train:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
        else:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
        
        self.data=x1+x2
        self.targets=y1+y2
        print(len(self.data),len(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets


class MiniImageNetJoint(MiniImageNet):
    """
    This is a subclass of the `MiniImageNet` Dataset customized for joint learning.
    """
    def SelectfromTxt(self, data2label, index_path):
        index=[]
        for txt_path in index_path:
            lines = [x.strip() for x in open(txt_path, 'r').readlines()]
            for line in lines:
                index.append(line.split('/')[3])
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.IMAGE_PATH, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from .autoaugment import AutoAugImageNetPolicy


class CUB200(Dataset):

    def __init__(self, root='/data', train=True, transform=None, index_path=None,
                 index=None, base_sess=None, autoaug=0, val=None, val_ratio=0, seed=None, img_size=224, is_feat_rect=False, savc_transform=None):
        self.root = os.path.expanduser(root)
        self.train = train # train set or validation/test set
        self.val = val # validation set or test set
        self.val_ratio = val_ratio
        self.seed = seed
        self._pre_operate(self.root)

        if train:
            if transform is not None:
                self.transform = transform
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    # transforms.CenterCrop(224),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            # self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
            if base_sess:
                # self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                self.data, self.targets = self.SelectfromClassesSplit(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxtSplit(self.data2label, index_path)
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            if self.val:
                if base_sess:
                    self.data, self.targets = self.SelectfromClassesSplit(self.data, self.targets, index)
                else:
                    self.data, self.targets = self.SelectfromTxtSplit(self.data2label, index_path)
            else:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
        
        if autoaug == 1:
            if train:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    # transforms.CenterCrop(224),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    #add autoaug
                    AutoAugImageNetPolicy(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                # self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
                if base_sess:
                    # self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
                    self.data, self.targets = self.SelectfromClassesSplit(self.data, self.targets, index)
                else:
                    self.data, self.targets = self.SelectfromTxtSplit(self.data2label, index_path)
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
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

    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines

    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict

    def _pre_operate(self, root):
        image_file = os.path.join(root, 'CUB_200_2011/images.txt')
        split_file = os.path.join(root, 'CUB_200_2011/train_test_split.txt')
        class_file = os.path.join(root, 'CUB_200_2011/image_class_labels.txt')
        id2image = self.list2dict(self.text_read(image_file))
        id2train = self.list2dict(self.text_read(split_file))  # 1: train images; 0: test iamges
        id2class = self.list2dict(self.text_read(class_file))
        train_idx = []
        test_idx = []
        for k in sorted(id2train.keys()):
            if id2train[k] == '1':
                train_idx.append(k)
            else:
                test_idx.append(k)

        self.data = []
        self.targets = []
        self.data2label = {}
        if self.train or self.val:
            for k in train_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

        else:
            for k in test_idx:
                image_path = os.path.join(root, 'CUB_200_2011/images', id2image[k])
                self.data.append(image_path)
                self.targets.append(int(id2class[k]) - 1)
                self.data2label[image_path] = (int(id2class[k]) - 1)

    def SelectfromTxt(self, data2label, index_path):
        index = open(index_path).read().splitlines()
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = os.path.join(self.root, i)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromTxtSplit(self, data2label, index_path):
        if isinstance(index_path, list):
            index = []
            for ip in index_path:
                index += open(ip).read().splitlines()
        else:
            index = open(index_path).read().splitlines()

        data_by_class = {}

        for i in index:
            img_path = os.path.join(self.root, i)
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


class CUB200_concate(Dataset):
    def __init__(self, train,x1,y1,x2,y2):
        
        if train:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.data=x1+x2
        self.targets=y1+y2
        print(len(self.data),len(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets


class CUB200Joint(CUB200):
    """
    This is a subclass of the `CUB200` Dataset customized for joint learning.
    """
    def SelectfromTxt(self, data2label, index_path):
        #index = open(index_path).read().splitlines()
        class_index = []
        for txt_path in index_path:
            with open(txt_path, 'r') as f:
                class_index.extend(f.read().splitlines()) 
        data_tmp = []
        targets_tmp = []

        for index in tqdm(class_index, "Make Dataloader"):
            img_path = os.path.join(self.root, index)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    
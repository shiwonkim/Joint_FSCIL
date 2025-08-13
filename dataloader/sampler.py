import torch
import numpy as np


class CategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per, ):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # sample n_cls classes from total classes.
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch
            # finally sample n_batch*  n_cls(way)* n_per(shot) instances. per bacth.


class BasePreserverCategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per, ):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            #classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # sample n_cls classes from total classes.
            classes=torch.arange(len(self.m_ind))
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch
            # finally sample n_batch*  n_cls(way)* n_per(shot) instances. per bacth.


class NewCategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per,):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        
        self.classlist=np.arange(np.min(label),np.max(label)+1)
        #print(self.classlist)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for c in self.classlist:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class PKsampler_base(torch.utils.data.Sampler):
    def __init__(self, dataset, p=64, k=8):
        self.p = p  # num samples
        self.k = k  # num classes
        self.dataset = dataset
        # self.num_classes = np.unique(self.dataset.targets).shape[0]
        self.all_classes = np.unique(self.dataset.targets)
        self.all_targets = self.dataset.targets
        self.batch_size = self.p * self.k

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        # choose target classes randomly
        num_batches = len(self.dataset) // self.batch_size
        res = []
        for i in range(num_batches):
            target_classes = np.random.choice(self.all_classes, self.k)
            for cls in target_classes:
                # choose samples
                cls_idx = np.where(self.all_targets == cls)[0]
                cls_idx = np.random.choice(cls_idx, self.p).tolist()
                res.append(cls_idx)
        res = np.concatenate(res).tolist()
        return iter(res)
    
class PKsampler_incr(torch.utils.data.Sampler):
    def __init__(self, dataset, p=64, k=8):
        self.p = p  # num samples
        self.k = k  # num classes
        self.dataset = dataset
        self.num_classes = self.dataset.num_cls
        self.batch_size = self.p * self.k

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        # choose target classes randomly
        num_batches = len(self.dataset) // self.batch_size
        res = []
        for i in range(num_batches):
            target_classes = torch.randperm(self.num_classes)[:self.k]
            res.append(torch.cat([target_classes for _ in range(self.p)]))
        res = torch.cat(res).tolist()
        return iter(res)


def get_weighted_sampler(args, cls_num_list, targets, cmo_flag):
    if cmo_flag is True:
        cls_weight = 1.0 / (np.array(cls_num_list) ** args.cmo_weighted_alpha)
    else :  cls_weight = 1.0 / np.array(cls_num_list)
    cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
    
    samples_weight = np.array([cls_weight[t] for t in targets])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    
    print("samples_weight", samples_weight)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(targets), replacement=True)
    
    return sampler
           

if __name__ == '__main__':
    q=np.arange(5,10)
    print(q)
    y=torch.tensor([5,6,7,8,9,5,6,7,8,9,5,6,7,8,9,5,5,5,55,])
    label = np.array(y)  # all data label
    m_ind = []  # the data index of each class
    for i in range(max(label) + 1):
        ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
        ind = torch.from_numpy(ind)
        m_ind.append(ind)
    print(m_ind, len(m_ind))

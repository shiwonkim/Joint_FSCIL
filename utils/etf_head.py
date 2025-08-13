import torch
import torch.nn as nn
import numpy as np


def reg_ETF(output, label, classifier, mse_loss):
    target = classifier.cur_M[:, label].T
    loss = mse_loss(output, target)
    
    return loss


def dot_loss(output, label, cur_M, criterion, H_length, reg_lam=0):
    target = cur_M[:, label].T
    
    if criterion == 'dot_loss':
        loss = -torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1).mean()
    elif criterion == 'reg_dot_loss':
        dot = torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1)

        with torch.no_grad():
            M_length = torch.sqrt(torch.sum(target ** 2, dim=1, keepdims=False))
        loss = (1/2) * torch.mean(((dot - (M_length * H_length)) ** 2) / H_length)

        if reg_lam > 0:
            reg_Eh_l2 = torch.mean(torch.sqrt(torch.sum(output ** 2, dim=1, keepdims=True)))
            loss = loss + reg_Eh_l2 * reg_lam

    return loss


def get_etf_matrix(args, classifier, target=None, mode='train', reg_Ew=True):
    if mode == 'train' and reg_Ew:
        learned_norm = produce_Ew(target, args.num_classes)

        if args.dataset == 'mini_imagenet':
            cur_M = learned_norm * classifier.module.ori_M
        else:
            cur_M = learned_norm * classifier.ori_M
    else:
        if args.dataset == 'mini_imagenet':
            cur_M = classifier.module.ori_M
        else:
            cur_M = classifier.ori_M

    return cur_M


def produce_Ew(label, num_classes):
    uni_label, count = torch.unique(label, return_counts=True)
    batch_size = label.size(0)
    uni_label_num = uni_label.size(0)
    
    assert batch_size == torch.sum(count)
    gamma = batch_size / uni_label_num
    Ew = torch.ones(1, num_classes).cuda(label.device)
    
    for i in range(uni_label_num):
        label_id = uni_label[i]
        label_count = count[i]
        length = torch.sqrt(gamma / label_count)
        Ew[0, label_id] = length
    
    return Ew


def produce_global_Ew(cls_num_list):
    num_classes = len(cls_num_list)
    cls_num_list = torch.tensor(cls_num_list).cuda()
    
    total_num = torch.sum(cls_num_list)
    gamma = total_num / num_classes
    
    Ew = torch.sqrt(gamma / cls_num_list)
    Ew = Ew.unsqueeze(0)
    
    return Ew


class ETF_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes, fix_bn=False, LWS=False, reg_ETF=False):
        super(ETF_Classifier, self).__init__()
        
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))
        self.ori_M = M.cuda()

        self.LWS = LWS
        self.reg_ETF = reg_ETF
        self.BN_H = nn.BatchNorm1d(feat_in)
        
        if fix_bn:
            self.BN_H.weight.requires_grad = False
            self.BN_H.bias.requires_grad = False

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        
        # Assert that feat_in >= num_classes
        error = torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07), error
        
        return P

    def forward(self, x):
        x = self.BN_H(x)
        x = x / torch.clamp(
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        
        return x

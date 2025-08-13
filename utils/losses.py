import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class SupContrastive(nn.Module):
    def __init__(self, reduction='mean'):
        super(SupContrastive, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        sum_neg = ((1 - y_true) * torch.exp(y_pred)).sum(1).unsqueeze(1)
        sum_pos = (y_true * torch.exp(-y_pred))
        num_pos = y_true.sum(1)
        loss = torch.log(1 + sum_neg * sum_pos).sum(1) / num_pos

        # For gradient clipping
        # Error occurs because of the hardware issue.
        # Reboot our computer can solve this problem.

        # sum_neg = ((1 - y_true) * torch.exp(torch.clamp(y_pred, max=10))).sum(1).unsqueeze(1)
        # sum_pos = (y_true * torch.exp(torch.clamp(-y_pred, min=-10)))
        # num_pos = y_true.sum(1) + 1e-6
        # loss = torch.log(1 + torch.clamp(sum_neg * sum_pos, min=1e-6)).sum(1) / num_pos

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list):
        super().__init__()
        cls_prior = cls_num_list / sum(cls_num_list)
        self.log_prior = torch.log(cls_prior).unsqueeze(0)
        self.reduction = 'mean'
        # self.min_prob = 1e-9
        # print(f'Use BalancedSoftmaxLoss, class_prior: {cls_prior}')

    def forward(self, logits, labels):
        adjusted_logits = logits + self.log_prior
        label_loss = F.cross_entropy(adjusted_logits, labels,
                                     reduction=self.reduction)

        return label_loss


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.reduction = "mean"

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target,
                               weight=self.weight, reduction=self.reduction)


class ImbSAM:
    def __init__(self, optimizer, model, rho=0.05):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.state = defaultdict(dict)

    @torch.no_grad()
    def first_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            
            grad_normal = self.state[p].get("grad_normal")
            if grad_normal is None:
                grad_normal = torch.clone(p).detach()
                self.state[p]["grad_normal"] = grad_normal
            
            grad_normal[...] = p.grad[...]
        
        self.optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
        
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
        
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        
        self.optimizer.zero_grad()

    @torch.no_grad()
    def third_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue

            p.sub_(self.state[p]["eps"])
            p.grad.add_(self.state[p]["grad_normal"])

        self.optimizer.step()
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()


def imbsam_loss(model, loss_func, optimizer, images,
                targets, logits, tail_classes, train_class, lam=None):
    loss_func.reduction = 'none'
    if isinstance(targets, list):
        loss = loss_func(logits, targets[0]) * lam + loss_func(logits, targets[1]) * (1-lam)
    else:
        loss = loss_func(logits, targets)
    
    if tail_classes is None:
        head_loss = loss.sum() / loss.size(0)
        head_loss.backward(retain_graph=True)
        optimizer.first_step()
        
        return head_loss

    if isinstance(targets, list):
        tail_mask = torch.where(((targets[0][:, None] == tail_classes[None, :].to(targets[0].device)).sum(1) == 1)
                                | ((targets[1][:, None] == tail_classes[None, :].to(targets[1].device)).sum(1) == 1), True, False)
    else:
        tail_mask = torch.where((targets[:, None] == tail_classes[None, :].to(targets.device)).sum(1) == 1, True, False)

    head_loss = loss[~tail_mask].sum() / tail_mask.size(0)
    head_loss.backward(retain_graph=True)
    optimizer.first_step()

    tail_loss = loss[tail_mask].sum() / tail_mask.size(0) * 2
    tail_loss.backward()
    optimizer.second_step()

    logits = model(images)[:, :train_class]
    if isinstance(targets, list):
        tail_loss = loss_func(logits[tail_mask], targets[0][tail_mask]) * lam + \
               loss_func(logits[tail_mask], targets[1][tail_mask]) * (1-lam)
    else:
        tail_loss = loss_func(logits[tail_mask], targets[tail_mask])
    tail_loss = tail_loss.sum() / tail_mask.size(0) * 2
    tail_loss.backward()
    optimizer.third_step()

    loss_func.reduction = 'mean'
    return head_loss + tail_loss


def etf_loss(args, dot_loss, logits, feat, train_label, cur_M, criterion, H_length,
             cmo=False, train_label2=None, lam=None):
    if isinstance(criterion, list) and args.imbsam:
        init_criterion = criterion[0]
    else:
        init_criterion = criterion

    if cmo:
        if isinstance(criterion, list) and not args.imbsam:
            loss_a = dot_loss(feat, train_label, cur_M, criterion[0], H_length, reg_lam=args.reg_lam)
            loss_b = dot_loss(feat, train_label2, cur_M, criterion[0], H_length, reg_lam=args.reg_lam)
            loss_1 = lam * loss_a + (1 - lam) * loss_b
            loss_2 = criterion[1](logits, train_label) * lam + criterion[1](logits, train_label2) * (1. - lam)
            loss = loss_1 + loss_2
        else:
            loss_a = dot_loss(feat, train_label, cur_M, init_criterion, H_length, reg_lam=args.reg_lam)
            loss_b = dot_loss(feat, train_label2, cur_M, init_criterion, H_length, reg_lam=args.reg_lam)
            loss = lam * loss_a + (1 - lam) * loss_b
    else:
        if isinstance(criterion, list) and not args.imbsam:
            loss_1 = dot_loss(feat, train_label, cur_M, criterion[0], H_length, reg_lam=args.reg_lam)
            loss_2 = criterion[1](logits, train_label)
            loss = loss_1 + loss_2
        else:
            loss = dot_loss(feat, train_label, cur_M, init_criterion, H_length, reg_lam=args.reg_lam)

    return loss

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss


def get_distance_matrix(x, type='distance', no_grad=False):
    if type == 'distance':
        d = pdist(x, squared=False)
        res = d
    else:
        assert type == 'angle'
        sd = (x.unsqueeze(0) - x.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        res = s_angle
    if no_grad:
        with torch.no_grad():
            res = res.detach()  # todo: necessary?
    return res


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            # mean_td = t_d[t_d > 0].mean()
            # t_d = t_d / mean_td

        d = pdist(student, squared=False)
        # mean_d = d[d > 0].mean()
        # d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss


def pairwise_kl_div_v2(logit_x, logit_y):
    """
    :param logit_x:
    :param logit_y:
    :return: p(x)*[log(p(x)) - log(p(y))]
    """
    log_px = F.log_softmax(logit_x, dim=1)
    log_py = F.log_softmax(logit_y, dim=1)
    px = F.softmax(logit_x, dim=1)

    px = px.unsqueeze(1).transpose(1, 2)
    logit = log_px.unsqueeze(1) - log_py.unsqueeze(0)
    kl_div = logit.bmm(px)
    return kl_div
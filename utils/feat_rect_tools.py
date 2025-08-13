# import new Network name here and add in model_class args
import torch.nn

from utils.tools import count_acc_per_task, Averager
from utils.feat_rect_tools import *
from tqdm import tqdm
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Distribution
from typing import Dict
from utils.losses import RkdDistance, RKdAngle, pairwise_kl_div_v2, get_distance_matrix
from utils.feat_rect_logger import LOGGER
from collections import defaultdict

log = LOGGER.LOGGER

def get_MVNs(data, label, base_label,
             encoder: torch.nn.Module,
             output_blocks,
             eps,
             MVN_distributions=None,
             k: int = 2):
    all_base_classes = sorted(torch.unique(base_label).tolist())
    all_novel_classes = sorted(torch.unique(label).tolist())
    feat, all_block_feats = encoder(data, return_all=True)
    _feat_mean = []
    for cls in all_novel_classes:
        cls_mask = label == cls
        cls_feat = feat[cls_mask]
        cls_feat_mean = cls_feat.mean(0)
        _feat_mean.append(cls_feat_mean)
    _feat_mean = torch.stack(_feat_mean)

    _means = [MVN_distributions[i]['strong_mean'] for i in all_base_classes]
    _cov = [MVN_distributions[i]['strong_cov'] for i in all_base_classes]
    _means = torch.stack(_means).cuda()
    _cov = torch.stack(_cov)
    dist_matrix = pairwise_cos_distance(_feat_mean, _means)  # B_new * B_novel
    selected_index = dist_matrix.argsort(-1)[:, :k].cpu()
    MVN_distributions = get_new_distributions(_cov, all_novel_classes, MVN_distributions,
                                              _feat_mean,
                                              selected_index,
                                              eps,
                                              prefix='strong_')
    for ii in output_blocks:
        _feat = all_block_feats[f'layer{ii}']
        _feat_mean = []
        for cls in all_novel_classes:
            cls_mask = label == cls
            cls_feat = _feat[cls_mask]
            cls_feat_mean = cls_feat.mean(0)
            _feat_mean.append(cls_feat_mean)
        _feat_mean = torch.stack(_feat_mean)
        _means = [MVN_distributions[i][f'layer{ii}_mean'] for i in all_base_classes]
        _cov = [MVN_distributions[i][f'layer{ii}_cov'] for i in all_base_classes]

        _means = torch.stack(_means).cuda()
        _cov = torch.stack(_cov)
        dist_matrix = pairwise_cos_distance(_feat_mean, _means)  # B_new * B_novel
        selected_index = dist_matrix.argsort(-1)[:, :k].cpu()
        MVN_distributions = get_new_distributions(_cov, all_novel_classes, MVN_distributions,
                                                  _feat_mean,
                                                  selected_index,
                                                  eps,
                                                  prefix=f'layer{ii}_')
    return MVN_distributions

def pairwise_cos_distance(features_a: torch.Tensor, features_b: torch.Tensor):
    # normalize two features
    n_features_a = F.normalize(features_a, dim=1)
    n_features_b = F.normalize(features_b, dim=1)
    # calculate Euclidean distance
    euc_dist = torch.norm(n_features_a[:, None] - n_features_b, p=2, dim=2)
    # calculate cosine distance ( 1- cos(a,b) )
    cos_dist = 0.5 * (torch.pow(euc_dist, 2))
    return cos_dist

def get_new_distributions(all_base_cov, all_novel_classes, distributions, mean_feats, selected_index, eps, prefix='', ):
    for idx, cls in enumerate(all_novel_classes):
        cls_selected_base_classes = selected_index[idx]
        new_mean = mean_feats[idx].cpu()

        new_cov = all_base_cov[cls_selected_base_classes].mean(0)

        tmp = torch.eye(len(new_cov)) * eps
        distribution_feat = MultivariateNormal(loc=new_mean, covariance_matrix=new_cov + tmp)

        cur_cls_dist = {f'{prefix}distribution': distribution_feat,
                        f'{prefix}mean': new_mean,
                        f'{prefix}origin_mean': mean_feats[idx],
                        f'{prefix}cov': new_cov}
        if cls not in distributions:
            distributions[cls] = cur_cls_dist
        else:
            distributions[cls].update(cur_cls_dist)

    return distributions
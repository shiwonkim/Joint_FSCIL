import torch
import random
import numpy as np
import pprint as pprint

from tqdm import tqdm
from torch.utils.data import DataLoader, Sampler
from collections import OrderedDict
from models.archs.warped_modules import *


class BatchSampler(Sampler):
    def __init__(self, dataset, num_iterations, batch_size):
        super().__init__(None)
        self.dataset = dataset
        self.num_iterations = num_iterations
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(self.num_iterations):
            indices = random.sample(range(len(self.dataset)), self.batch_size)
            yield indices

    def __len__(self):
        return self.num_iterations


def compute_orthonormal(args, net, trainset):
    training = net.training
    indices = torch.randperm(len(trainset))[:20 * args.batch_size_base]


    trainset = torch.utils.data.Subset(trainset, indices)
    dl = DataLoader(trainset, shuffle=False, batch_size=args.batch_size_base)
    net.eval()
    net.forward_covariance = None
    net.batch_count = 0

    cdmodule_list = [module for module in net.modules() if isinstance(module, WaRPModule)]
    for m in cdmodule_list:
        m.flag = False
    epoch_iter = tqdm(dl)

    with torch.no_grad():
        for x, y in epoch_iter:
            x, y = x.cuda(), y.cuda()
            encoded_x = net.module.encode(x)[:, :args.base_class]
            for m in cdmodule_list:
                m.post_backward()

        module_it = tqdm(cdmodule_list)
        for m in module_it:
            m.flag = True
            forward_cov = getattr(m, 'forward_covariance')

            V, S, UT_forward = torch.linalg.svd(forward_cov, full_matrices=True)

            weight = getattr(m, 'weight')
            bias = getattr(m, 'bias')
            if weight.ndim != 2:
                weight = weight.reshape(weight.shape[0], -1)

            UT_backward = torch.eye(weight.shape[0])
            UT_backward = same_device(ensure_tensor(UT_backward), weight)


            UT_forward = same_device(UT_forward, weight)
            basis_coefficients = UT_backward @ weight @ UT_forward.t()
            m.UT_forward = UT_forward
            m.UT_backward = UT_backward
            m.basis_coefficients.data = basis_coefficients.data
            m.basis_coeff.data = basis_coefficients.data

        for m in module_it:

            coefficients = getattr(m, 'basis_coefficients')
            UT_forward = getattr(m, 'UT_forward')
            UT_backward = getattr(m, 'UT_backward')
            if m.weight.ndim != 2:
                basis_coefficients = coefficients.reshape(m.weight.shape[0],
                                                    UT_forward.shape[0],
                                                    1, 1)
                m.UT_forward_conv = UT_forward.reshape(UT_forward.shape[0],
                                                               m.weight.shape[1],
                                                               m.weight.shape[2],
                                                               m.weight.shape[3])
                m.UT_backward_conv = UT_backward.t().reshape(m.weight.shape[0],
                                                                     m.weight.shape[0],
                                                                     1,
                                                                     1)
                m.basis_coeff.data = basis_coefficients.data

    net.training = training


def identify_importance(args, model, trainset, batchsize=60, keep_ratio=0.1, session=0, way=10, new_labels=None):
    importances = OrderedDict()
    temp = OrderedDict()
    dl = DataLoader(trainset, shuffle=False, batch_size=batchsize)
    model.eval().cuda()

    for module in model.modules():
        if isinstance(module, WaRPModule):

            module.coeff_mask_prev = module.coeff_mask.data
            module.coeff_mask.data = torch.zeros(module.coeff_mask.shape).cuda().data

    training = model.training

    epoch_iter = tqdm(dl)
    for i, batch in enumerate(epoch_iter):
        if session == 0:
            x, y = [_.cuda() for _ in batch]
        else:
            x, y = batch.cuda(), new_labels.cuda()[i * batchsize:(i+1) * batchsize]
        yhat = model(x)[:, :args.base_class + session * way]
        loss = nn.CrossEntropyLoss()(yhat, y)
        model.zero_grad()
        loss.backward()

        for module in model.modules():
            if isinstance(module, WaRPModule):
                temp[module] = module.basis_coeff.grad.abs().detach().cpu().numpy().copy()

        for module in model.modules():
            if isinstance(module, WaRPModule):
                if module not in importances:
                    importances[module] = temp[module]
                else:
                    importances[module] += temp[module]


    flat_importances = flatten_importances_module(importances)
    threshold = fraction_threshold(flat_importances, keep_ratio)
    masks = importance_masks_module(importances, threshold)


    for module in model.modules():
        if isinstance(module, WaRPModule):
            coeff_mask = masks[module]
            coeff_mask = same_device(ensure_tensor(coeff_mask), module.basis_coefficients)
            module.coeff_mask.data = 1 - (1 - coeff_mask.data) * (1 - module.coeff_mask_prev.data)


    # -------------------------- get accumulative mask ratio ---------------------------------
    for module in model.modules():
        if isinstance(module, WaRPModule):
            masks[module] = module.coeff_mask.data.detach().cpu().numpy().copy()
    print(flatten_importances_module(masks).mean())
    # ----------------------------------------------------------------------------------------


    model.zero_grad()
    model.training = training

    for module in model.modules():
        if hasattr(module, 'weight') and not isinstance(module, WaRPModule):
            for param in module.parameters():
                param.requires_grad = False

    return model





def flatten_importances_module(importances):
    return np.concatenate([
        params.flatten()
        for _, params in importances.items()
    ])

def map_importances_module_dict(fn, importances):
    return {module: fn(params)
            for module, params in importances.items()}

def importance_masks_module(importances, threshold):
    return map_importances_module_dict(lambda imp: threshold_mask(imp, threshold), importances)


def fraction_threshold(tensor, fraction):
    """Compute threshold quantile for a given scoring function

    Given a tensor and a fraction of parameters to keep,
    computes the quantile so that only the specified fraction
    are larger than said threshold after applying a given scoring
    function. By default, magnitude pruning is applied so absolute value
    is used.

    Arguments:
        tensor {numpy.ndarray} -- Tensor to compute threshold for
        fraction {float} -- Fraction of parameters to keep

    Returns:
        float -- Threshold
    """
    assert isinstance(tensor, np.ndarray)
    threshold = np.quantile(tensor, 1-fraction)
    return threshold

def threshold_mask(tensor, threshold):
    """Given a fraction or threshold, compute binary mask

    Arguments:
        tensor {numpy.ndarray} -- Array to compute the mask for

    Keyword Arguments:
        threshold {float} -- Absolute threshold for dropping params

    Returns:
        np.ndarray -- Binary mask
    """
    assert isinstance(tensor, np.ndarray)
    idx = np.logical_and(tensor < threshold, tensor > -threshold)
    mask = np.ones_like(tensor)
    mask[idx] = 0
    return mask




def restore_weight(net):
    cdmodule_list = [module for module in net.modules() if isinstance(module, WaRPModule)]
    for module in cdmodule_list:
        weight = getattr(module, 'weight')
        UT_forward = getattr(module, 'UT_forward')
        UT_backward = getattr(module, 'UT_backward')
        coeff_mask = getattr(module, 'coeff_mask').reshape(weight.shape[0], -1)

        coeff_mask = same_device(ensure_tensor(coeff_mask), module.basis_coeff.data)
        weight_res = UT_backward.t() @ module.basis_coeff.data.reshape(coeff_mask.shape) @ UT_forward
        weight_res = weight_res.reshape(weight.shape)
        module.weight.data = weight_res.data
    return net


def compute_accum_ratio(model, session):
    masks = OrderedDict()
    for module in model.modules():
        if isinstance(module, WaRPModule):
            masks[module] = module.coeff_mask.data.detach().cpu().numpy().copy()
    logs = dict(num_session=session, keep_ratio=flatten_importances_module(masks).mean().item())
    return logs

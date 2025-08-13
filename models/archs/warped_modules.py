import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = F.pad(input_data, [pad, pad, pad, pad], 'constant', 0)
    col = torch.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = torch.permute(col, (0, 4, 5, 1, 2, 3)).reshape(N * out_h * out_w, -1)
    return col


def im2col_from_conv(input_data, conv):
    return im2col(input_data, conv.kernel_size[0], conv.kernel_size[1], conv.stride[0], conv.padding[0])


def get_params(model, recurse=False):
    """Returns dictionary of paramters

    Arguments:
        model {torch.nn.Module} -- Network to extract the parameters from

    Keyword Arguments:
        recurse {bool} -- Whether to recurse through children modules

    Returns:
        Dict(str:numpy.ndarray) -- Dictionary of named parameters their
                                   associated parameter arrays
    """
    params = {k: v.detach().cpu().numpy().copy()
              for k, v in model.named_parameters(recurse=recurse)}
    return params


def get_buffers(model, recurse=False):
    """Returns dictionary of buffers

    Arguments:
        model {torch.nn.Module} -- Network to extract the buffers from

    Keyword Arguments:
        recurse {bool} -- Whether to recurse through children modules

    Returns:
        Dict(str:numpy.ndarray) -- Dictionary of named parameters their
                                   associated parameter arrays
    """
    buffers = {k: v.detach().cpu().numpy().copy()
              for k, v in model.named_buffers(recurse=recurse)}
    return buffers


def ensure_tensor(x):
    if not isinstance(x, torch.Tensor) and x is not None:
        return torch.from_numpy(x)
    return x


def same_device(x_mask, x):
    if x.device != x_mask.device:
        return x_mask.to(x.device)
    return x_mask


def _same_shape(x_mask, x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    return x.shape == x_mask.shape


class WaRPModule(nn.Module):
    def __init__(self, layer):
        super(WaRPModule, self).__init__()

        self.weight = layer.weight
        self.bias = layer.bias
        
        if self.weight.ndim != 2:
            Co, Ci, k1, k2 = self.weight.shape
            self.basis_coeff = nn.Parameter(torch.Tensor(Co, Ci*k1*k2, 1, 1), requires_grad=True)
            self.register_buffer("UT_forward_conv", torch.Tensor(Ci*k1*k2, Ci, k1, k2))
            self.register_buffer("UT_backward_conv", torch.Tensor(Co, Co, 1, 1))
        else:
            self.basis_coeff = nn.Parameter(torch.Tensor(self.weight.shape), requires_grad=True)


        # use register_buffer so model.to(device) works on fixed tensors like masks
        self.register_buffer("forward_covariance", None)
        self.register_buffer("basis_coefficients", torch.Tensor(self.weight.shape).reshape(self.weight.shape[0], -1))
        self.register_buffer("coeff_mask", torch.zeros(self.basis_coeff.shape))
        self.register_buffer("UT_forward", torch.eye(self.basis_coeff.shape[1]))
        self.register_buffer("UT_backward", torch.eye(self.basis_coeff.shape[0]))

        self.flag = True


class LinearWaRP(WaRPModule):
    def __init__(self, linear_layer):
        super(LinearWaRP, self).__init__(linear_layer)
        assert isinstance(linear_layer, nn.Linear), "Layer must be a linear layer"
        for attr in ['in_features', 'out_features']:
            setattr(self, attr, getattr(linear_layer, attr))

        self.batch_count = 0

    def pre_forward(self, input):
        with torch.no_grad():
            if self.bias is not None:
                pass
            forward_covariance = input.t() @ input
        return forward_covariance

    def post_forward(self, input):
        self.h = input.register_hook(set_grad(input))
        return input

    def post_backward(self):
        with torch.no_grad():
            if self.forward_covariance is not None:
                self.forward_covariance = self.forward_curr_cov + (self.batch_count / (self.batch_count + 1)) * \
                (self.forward_covariance - self.forward_curr_cov)
            else:
                self.forward_covariance = self.forward_curr_cov


            self.batch_count += 1

    def forward(self, input):
        if not self.flag:
            self.forward_curr_cov = self.pre_forward(input)
            input = F.linear(input, self.weight, self.bias)
        else:
            weight = self.UT_backward @ (self.basis_coeff * self.coeff_mask).clone().detach() + self.basis_coeff * (
                        1 - self.coeff_mask) @ self.UT_forward
            input = F.linear(input, weight)

        return input

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f'in_features={self.in_features}, '
        s += f'out_features={self.out_features}, '
        s += f'bias={self.bias is not None})'
        return s


class Conv2dWaRP(WaRPModule):
    def __init__(self, conv_layer):
        super(Conv2dWaRP, self).__init__(conv_layer)
        assert isinstance(conv_layer, nn.Conv2d), "Layer must be a Conv2d layer"
        for attr in ['in_channels', 'out_channels', 'kernel_size', 'dilation',
                     'stride', 'padding', 'padding_mode', 'groups']:
            setattr(self, attr, getattr(conv_layer, attr))

        self.batch_count = 0

    def pre_forward(self, input):
        with torch.no_grad():
            input_col = im2col_from_conv(input.clone(), self)
            forward_covariance = input_col.t() @ input_col
        return forward_covariance

    def post_forward(self, input):
        self.h = input.register_hook(set_grad(input))
        return input

    def post_backward(self):
        with torch.no_grad():
            if self.forward_covariance is not None:
                self.forward_covariance = self.forward_cov + (self.batch_count / (self.batch_count + 1)) * \
                                          (self.forward_covariance - self.forward_cov)
            else:
                self.forward_covariance = self.forward_cov

            self.batch_count += 1

    def forward(self, input):
        if not self.flag:
            self.forward_cov = self.pre_forward(input)
            if self.padding_mode == 'circular':
                expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                    (self.padding[0] + 1) // 2, self.padding[0] // 2)
                input = F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                                self.weight, self.bias, self.stride,
                                _pair(0), self.dilation, self.groups)
            else:
                input = F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        else:
            UTx = F.conv2d(input, self.UT_forward_conv, None, self.stride,
                           self.padding, self.dilation, self.groups)
            AUTx = F.conv2d(UTx, (self.basis_coeff * self.coeff_mask).clone().detach() + self.basis_coeff * (
                        1 - self.coeff_mask), None, 1, 0)
            input = F.conv2d(AUTx, self.UT_backward_conv, self.bias, 1, 0)

        return input

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
              ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(**self.__dict__)


warped_modules = {
    nn.Linear: LinearWaRP,
    nn.Conv2d: Conv2dWaRP,
}


def switch_module(module):
    new_children = {}

    for name, submodule in module.named_children():
        if isinstance(submodule, nn.Conv2d) and hasattr(submodule, 'is_warp_conv'):

            switched = warped_modules[type(submodule)](submodule)
            new_children[name] = switched

        switch_module(submodule)

    for name, switched in new_children.items():
        setattr(module, name, switched)

    return module.cuda()

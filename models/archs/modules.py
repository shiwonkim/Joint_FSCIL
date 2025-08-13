import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import Parameter
from timm.models.layers import to_2tuple


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n * b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n * b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n * b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n * dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class StochasticClassifier(nn.Module):
    def __init__(self, num_features, num_classes, temp):
        super().__init__()

        self.weight = nn.Parameter(0.01 * torch.randn(num_classes, num_features))
        self.sigma = nn.Parameter(torch.zeros(num_classes, num_features)) # each rotation have individual variance here
        self.temp = temp
    
    def forward(self, x, stochastic=True):
        mu = self.weight
        sigma = self.sigma

        if stochastic:
            sigma = F.softplus(sigma - 4) # when sigma=0, softplus(sigma-4)=0.0181
            weight = sigma * torch.randn_like(mu) + mu
        else:
            weight = mu
        
        weight = F.normalize(weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        score = F.linear(x, weight)
        score = score * self.temp

        return score


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        # Convolutional filters, work excellent with image data
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=False),

            # 3D and 32 by 32
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 1, 0, bias=False),

            nn.BatchNorm2d(self.dim_h * 8), # 40 X 8 = 320
            # nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=False))
        
        # nn.Conv2d(self.dim_h * 8, 1, 2, 1, 0, bias=False))
        # nn.Conv2d(self.dim_h * 8, 1, 4, 1, 0, bias=False))
        
        # Final layer is fully connected
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze()
        x = self.fc(x)

        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        # First layer is fully connected
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
            nn.ReLU(False))

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, self.n_channel, 4, stride=2),
            nn.Tanh())

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.deconv(x)
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        
        return x
    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])
        self.norm = nn.LayerNorm(hidden_features, eps=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class FeatureRectify(nn.Module):
    def __init__(self, feature_dim, act_layer=nn.GELU, drop=0.):
        super(FeatureRectify, self).__init__()
        self.mlp_inter = Mlp(feature_dim, hidden_features=512, out_features=feature_dim, act_layer=act_layer, drop=drop)
        self.mlp_final = Mlp(feature_dim, hidden_features=512, out_features=feature_dim, act_layer=act_layer, drop=drop)
        self.mlp_mixed = Mlp(feature_dim * 2, out_features=feature_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_inter, x_final):
        x_inter = self.mlp_inter(x_inter)
        x_final = self.mlp_final(x_final)
        x = torch.cat((x_inter, x_final), dim=1)
        res = self.mlp_mixed(x)
        return res

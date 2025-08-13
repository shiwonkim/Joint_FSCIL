import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.archs.resnets.resnet18_encoder import *
from models.archs.resnets.resnet20_cifar import *
from models.archs.resnets.warped.resnet18_encoder import resnet18_warped
from models.archs.resnets.warped.resnet20_cifar import resnet20_warped

from models.archs.modules import *
from utils.etf_head import ETF_Classifier
from utils.warped_tools import identify_importance

from models.archs.vits.deit import deit_my_small_patch3_MultiLyaerOutput, \
    deit_my_small_patch8_MultiLyaerOutput, deit_my_small_patch16_MultiLyaerOutput_224
from tqdm import tqdm


class BaseNet(nn.Module):
    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args

        if self.args.dataset in ['cifar100']:
            self.encoder = resnet20()
            self.num_features = 128
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet18(False, args)
            self.num_features = 512
        if self.args.dataset in ['cub200']:
            self.encoder = resnet18(True, args) # Follow TOPIC: use imagenet pre-trained model for CUB
            self.num_features = 512
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)

    def forward(self, input):
        if self.mode == 'encoder':
            input = self.encode(input)
            return input
        elif self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        else:
            raise ValueError('Unknown mode')
        
    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        
        return x
    
    def forward_metric(self, x):
        x = self.encode(x)
    
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        
        return x
    
    def update_fc(self, dataloader, class_list, session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data = self.encode(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"), requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode: # further finetune
            self.update_fc_ft(new_fc, data, label, session)

    def update_fc_avg(self, data, label, class_list):
        new_fc = []

        for class_index in class_list:
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[data_index]
            proto = embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index] = proto
        
        new_fc = torch.stack(new_fc, dim=0)
        
        return new_fc

    def get_logits(self, x, fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x, fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self, new_fc, data, label, session):
        new_fc = new_fc.clone().detach()
        new_fc.requires_grad = True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.lr_new,
                                    momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data, fc)
                loss = F.cross_entropy(logits, label)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass
        
        start_idx = self.args.base_class + self.args.way * (session - 1)
        end_idx = self.args.base_class + self.args.way * session
        self.fc.weight.data[start_idx:end_idx, :].copy_(new_fc.data)


class BaseNet_ViT(nn.Module):
    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args
        self.branches = 1
        
        if self.args.dataset in ['cifar100']:
            self.encoder = deit_my_small_patch3_MultiLyaerOutput(num_classes=0, output_blocks=args.output_blocks,
                                                                 image_size=args.image_size)
            self.num_features = self.encoder.num_features
        elif self.args.dataset in ['mini_imagenet']:
            self.encoder = deit_my_small_patch8_MultiLyaerOutput(num_classes=0, output_blocks=args.output_blocks,
                                                                 image_size=args.image_size)
            self.num_features = self.encoder.num_features
        elif self.args.dataset in ['cub200']:
            self.encoder = self.encoder = deit_my_small_patch16_MultiLyaerOutput_224(num_classes=0, output_blocks=args.output_blocks,
                                                                 image_size=args.image_size)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = self.encoder.num_features
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))


class CECNet(BaseNet):
    def __init__(self, args, mode=None):
        super().__init__(args, mode)
        
        hdim = self.num_features
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)

    def _forward(self, support, query):
        ''' For pseudo incremental learning '''
        
        emb_dim = support.size(-1)
        
        # Get the mean of the support
        proto = support.mean(dim=1)
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = query.shape[1]*query.shape[2] # num of query * way

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        query = query.view(-1, emb_dim).unsqueeze(1)

        proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
        proto = proto.view(num_batch * num_query, num_proto, emb_dim)

        combined = torch.cat([proto, query], 1) # Nk x (N + 1) x d, batch_size = NK
        combined = self.slf_attn(combined, combined, combined)
        
        # Compute the distance for all batches
        proto, query = combined.split(num_proto, 1)

        logits = F.cosine_similarity(query, proto, dim=-1)
        logits = logits * self.args.temperature

        return logits


class FACTNet(BaseNet):
    def __init__(self, args, mode=None):
        super().__init__(args, mode)
        
        self.pre_allocate = self.args.num_classes
        self.fc = nn.Linear(self.num_features, self.pre_allocate, bias=False)
        
        nn.init.orthogonal_(self.fc.weight)
        self.dummy_orthogonal_classifier = nn.Linear(self.num_features, self.pre_allocate-self.args.base_class, bias=False)
        self.dummy_orthogonal_classifier.weight.requires_grad = False
        
        self.dummy_orthogonal_classifier.weight.data = self.fc.weight.data[self.args.base_class:, :]
        print(self.dummy_orthogonal_classifier.weight.data.size())

    def forward_metric(self, x):
        x = self.encode(x)
        
        if 'cos' in self.mode:
            x1 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x2 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.dummy_orthogonal_classifier.weight, p=2, dim=-1))
        
            x = torch.cat([x1[:, :self.args.base_class], x2], dim=1)
            x = self.args.temperature * x
        
        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        
        return x

    def forpass_fc(self,x):
        x = self.encode(x)
        
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x
            
        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        
        return x
    
    def pre_encode(self,x):
        if self.args.dataset in ['cifar100']:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
            
        elif self.args.dataset in ['cub200', 'mini_imagenet']:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
        
        return x
        
    def post_encode(self,x):
        if self.args.dataset in ['cifar100']:
            x = self.encoder.layer3(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)

        elif self.args.dataset in ['cub200', 'mini_imagenet']:
            x = self.encoder.layer3(x)
            x = self.encoder.layer4(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)
        
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
            
        return x


class TEENNet(BaseNet):
    def __init__(self, args, mode=None):
        super().__init__(args, mode)

    def soft_calibration(self, args, session):
        base_protos = self.fc.weight.data[:args.base_class].detach().cpu().data
        base_protos = F.normalize(base_protos, p=2, dim=-1)
        
        start_idx = args.base_class + args.way * (session - 1)
        end_idx = args.base_class + args.way * session
        cur_protos = self.fc.weight.data[start_idx:end_idx].detach().cpu().data
        cur_protos = F.normalize(cur_protos, p=2, dim=-1)
        
        weights = torch.mm(cur_protos, base_protos.T) * args.softmax_t
        norm_weights = torch.softmax(weights, dim=1)
        delta_protos = torch.matmul(norm_weights, base_protos)
        delta_protos = F.normalize(delta_protos, p=2, dim=-1)
        
        updated_protos = (1 - args.shift_weight) * cur_protos + args.shift_weight * delta_protos
        self.fc.weight.data[start_idx:end_idx] = updated_protos


class FEAT_RECTNet(BaseNet_ViT):
    def __init__(self, args, mode=None):
        super().__init__(args, mode=None)

        self.fc = nn.ModuleDict({f'layer{i}': nn.Linear(self.num_features, self.args.num_classes, bias=False) for i in
                                 self.args.output_blocks})  # use classifier for each blocks
        self.fc['final'] = nn.Linear(self.num_features, self.args.num_classes, bias=False)

        # move checkpoint loading to main func

        self.feature_rectification = nn.ModuleDict({f'layer{i}': FeatureRectify(self.encoder.num_features) for i in
                                                    self.args.output_blocks})
    def forward(self, x, fc_type='fc', return_all_feats=False):
        x = self.encoder(x, return_all=True)
        all_feats = x['output_features_per_block']
        x = x['features']
        if self.mode != 'encoder':
            assert False
        elif self.mode == 'encoder':
            if return_all_feats:
                return x, all_feats
            else:
                return x
        else:
            raise ValueError('Unknown mode')

    def update_fc(self, dataloader, encoder, output_blocks):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            with torch.no_grad():
                strong_feat, all_feats = encoder(data, return_all=True)

        self.update_fc_avg(self.fc, strong_feat, all_feats, label, np.unique(label.detach().cpu()), output_blocks)

    def update_fc_avg(self, fc, strong_feature, all_feature, label, class_list, output_blocks):
        for class_index in class_list:
            data_index = (label == class_index).nonzero().squeeze(-1)
            embedding = strong_feature[data_index]
            proto = embedding.mean(0)
            proto = F.normalize(proto, p=2, dim=-1)
            fc['final'].weight.data[class_index] = proto

            for ii in output_blocks:
                embedding = all_feature[f'layer{ii}'][data_index]
                proto = embedding.mean(0)
                proto = F.normalize(proto, p=2, dim=-1)
                fc[f'layer{ii}'].weight.data[class_index] = proto

    def update_fc_ft(self, new_fc, data, label, session, branches=1, MVN_distributions=None, dataloader=None):
        return
    
    def extract_features(self, encoder, dataloader, args, mapping=None):
        all_labels = []
        final_features = torch.tensor([])
        final_mapped = {f'layer{i}': torch.tensor([]).cuda() for i in args.output_blocks}
        all_features = {f'layer{i}': torch.tensor([]) for i in args.output_blocks}
        with torch.no_grad():
            tqdm_gen = tqdm(dataloader)
            for i, batch in enumerate(tqdm_gen, 1):
                data, test_label = [_.cuda() for _ in batch]
                bs, c, h, w = data.shape
                feat, all_block_features = encoder(data, return_all=True)
                _feat = feat.cpu()
                final_features = torch.cat((final_features, _feat))
                for ii in args.output_blocks:
                    if mapping is not None:
                        _map = mapping[f'layer{ii}']
                        inter_feat = all_block_features[f'layer{ii}']
                        feat_mapped = _map(inter_feat, feat)
                        final_mapped[f'layer{ii}'] = torch.cat((final_mapped[f'layer{ii}'], feat_mapped))
                    cur_feat = all_block_features[f'layer{ii}'].cpu()
                    all_features[f'layer{ii}'] = torch.cat((all_features[f'layer{ii}'], cur_feat))
                all_labels.append(test_label.cpu())
        if mapping is not None:
            return final_features, all_features, torch.cat(all_labels), final_mapped
        return final_features, all_features, torch.cat(all_labels)


class S3CNet(BaseNet):
    def __init__(self, args, mode=None):
        super().__init__(args, mode)
        
        self.fc = StochasticClassifier(self.num_features, self.args.num_classes*4, self.args.temperature)
        hdim = self.num_features
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)

    def forward(self, input, stochastic=True):
        if self.mode != 'encoder':
            input = self.forward_metric(input, stochastic)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def forward_metric(self, x, stochastic=True):
        x_f, x_f_a = self.encode(x)
        
        if 'cos' in self.mode:
            x = self.fc(x_f, stochastic)
        elif 'dot' in self.mode:
            x = self.fc(x_f)

        return x, x_f, x_f_a

    def forward_proto(self, x, stochastic=True):
        x = x.unsqueeze(1)
        x = self.slf_attn(x, x, x)
        x = x.squeeze(1)
        x = self.fc(x, stochastic)
        
        return x

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x_f_a = x.squeeze(-1).squeeze(-1)
        
        x = x_f_a.unsqueeze(1)
        x = self.slf_attn(x, x, x)
        x = x.squeeze(1)
       
        return x, x_f_a

    def update_fc(self, dataloader, class_list, session):
        class_list = torch.from_numpy(class_list)
        class_list = torch.stack([class_list * 4 + k for k in range(4)], 1).view(-1)
        
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data = torch.stack([torch.rot90(data, k, (2, 3)) for k in range(4)], 1)
            data = data.view(-1, 3, 32, 32)
            label = torch.stack([label * 4 + k for k in range(4)], 1).view(-1)
            data, _=self.encode(data)
            data = data.detach()

        if self.args.not_data_init_new:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:
            self.update_fc_ft(new_fc, data, label, session)


class WaRPNet(BaseNet):
    def __init__(self, args, mode=None):
        super().__init__(args, mode)
        
        self.mode = mode
        self.args = args

        if self.args.dataset in ['cifar100']:
            self.encoder = resnet20_warped()
            self.num_features = 128
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet18_warped(False, args)
            self.num_features = 512
        if self.args.dataset in ['cub200']:
            self.encoder = resnet18_warped(True, args)
            self.num_features = 512
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.fc.is_classifier = True

    def forward_metric(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)

        else:
            base_logits = self.fc(x)[:, :self.args.base_class]
            new_logits = - (torch.unsqueeze(self.fc.weight.t(), dim=0) - torch.unsqueeze(x, dim=-1)).norm(p=2, dim=1)[:, self.args.base_class:]
            x = torch.cat([base_logits, new_logits], dim=-1)

        return x

    def update_fc(self, dataloader, class_list, session):
        global data_imgs
        
        for batch in dataloader:
            data_imgs, label = [_.cuda() for _ in batch]
            data = self.encode(data_imgs).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"), requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:
            self.update_fc_ft(data_imgs, label, session)

    def get_logits(self, x, fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x, fc)

        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

        else:
            base_logits = F.linear(x, fc)[:, :self.args.base_class]
            new_logits = - (torch.unsqueeze(fc.t(), dim=0) - torch.unsqueeze(x, dim=-1)).norm(p=2, dim=1)[:, self.args.base_class:]
            return torch.cat([base_logits, new_logits], dim=-1)

    def update_fc_ft(self, data_imgs, label, session):
        self.eval()
        optimizer_embedding = torch.optim.SGD(self.encoder.parameters(), lr=self.args.lr_new, momentum=0.9)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                fc = self.fc.weight[:self.args.base_class + self.args.way * session, :].detach()
                data = self.encode(data_imgs)
                logits = self.get_logits(data, fc)

                loss = F.cross_entropy(logits, label)
                optimizer_embedding.zero_grad()
                loss.backward()

                optimizer_embedding.step()

        identify_importance(self.args, self, data_imgs.cpu(), batchsize=1, keep_ratio=self.args.fraction_to_keep,
                            session=session, way=self.args.way, new_labels=label)


class SAVCNet(BaseNet):
    def __init__(self, args, mode=None, trans=1):
        super().__init__(args, mode)

        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar100']:
            self.encoder_q = resnet20(num_classes=self.args.moco_dim, return_logits=True)
            self.encoder_k = resnet20(num_classes=self.args.moco_dim, return_logits=True)
            self.num_features = 128
        if self.args.dataset in ['mini_imagenet']:
            self.encoder_q = resnet18(False, args, num_classes=self.args.moco_dim, return_logits=True)  # pretrained=False
            self.encoder_k = resnet18(False, args, num_classes=self.args.moco_dim, return_logits=True)  # pretrained=False
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder_q = resnet18(True, args, num_classes=self.args.moco_dim, return_logits=True)
            self.encoder_k = resnet18(True, args,
                                      num_classes=self.args.moco_dim, return_logits=True)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.args.num_classes * trans, bias=False)

        self.K = self.args.moco_k
        self.m = self.args.moco_m
        self.T = self.args.moco_t

        if self.args.mlp:  # hack: brute-force replacement
            self.encoder_q.fc = nn.Sequential(nn.Linear(self.num_features, self.num_features), nn.ReLU(),
                                              self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(self.num_features, self.num_features), nn.ReLU(),
                                              self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(self.args.moco_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("label_queue", torch.zeros(self.K).long() - 1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self, base_sess):
        """
        Momentum update of the key encoder
        """
        if base_sess:
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
                # print("Momentum Update - param_k min: {}, max: {}".format(param_k.min(), param_k.max()))
        else:
            for k, v in self.encoder_q.named_parameters():
                if k.startswith('fc') or k.startswith('layer4') or k.startswith('layer3'):
                    self.encoder_k.state_dict()[k].data = self.encoder_k.state_dict()[k].data * self.m + v.data * (
                                1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # replace the keys and labels at ptr (dequeue and enqueue)
        if ptr + batch_size > self.K:
            remains = ptr + batch_size - self.K
            self.queue[:, ptr:] = keys.T[:, :batch_size - remains]
            self.queue[:, :remains] = keys.T[:, batch_size - remains:]
            self.label_queue[ptr:] = labels[:batch_size - remains]
            self.label_queue[:remains] = labels[batch_size - remains:]
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T  # this queue is feature queue
            self.label_queue[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def forward_metric(self, x):
        x, _ = self.encode_q(x)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)

        return x  # joint, contrastive

    def encode_q(self, x):
        x, y = self.encoder_q(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x, y

    def encode_k(self, x):
        x, y = self.encoder_k(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x, y

    def forward(self, im_cla, im_q=None, im_k=None, labels=None,
                im_q_small=None, base_sess=True, last_epochs_new=False):
        if self.mode != 'encoder':
            if im_q == None:
                x = self.forward_metric(im_cla)
                return x
            else:
                b = im_q.shape[0]
                logits_classify = self.forward_metric(im_cla)
                _, q = self.encode_q(im_q)

                q = nn.functional.normalize(q, dim=1)
                feat_dim = q.shape[-1]
                q = q.unsqueeze(1)  # bs x 1 x dim

                if im_q_small is not None:
                    _, q_small = self.encode_q(im_q_small)
                    q_small = q_small.view(b, -1, feat_dim)  # bs x 4 x dim
                    q_small = nn.functional.normalize(q_small, dim=-1)

                # compute key features
                with torch.no_grad():  # no gradient to keys
                    self._momentum_update_key_encoder(base_sess)  # update the key encoder
                    _, k = self.encode_k(im_k)  # keys: bs x dim
                    k = nn.functional.normalize(k, dim=1)

                # compute logits
                # Einstein sum is more intuitive
                # positive logits: Nx1
                q_global = q
                l_pos_global = (q_global * k.unsqueeze(1)).sum(2).view(-1, 1)
                l_pos_small = (q_small * k.unsqueeze(1)).sum(2).view(-1, 1)

                # negative logits: NxK
                l_neg_global = torch.einsum('nc,ck->nk', [q_global.view(-1, feat_dim), self.queue.clone().detach()])
                l_neg_small = torch.einsum('nc,ck->nk', [q_small.view(-1, feat_dim), self.queue.clone().detach()])
                # print("Queue Min: {}, Max: {}".format(self.queue.min(), self.queue.max()))

                # logits: Nx(1+K)
                logits_global = torch.cat([l_pos_global, l_neg_global], dim=1)
                logits_small = torch.cat([l_pos_small, l_neg_small], dim=1)

                # apply temperature
                logits_global /= self.T
                logits_small /= self.T

                # one-hot target from augmented image
                positive_target = torch.ones((b, 1)).cuda()
                # find same label images from label queue
                # for the query with -1, all
                targets = ((labels[:, None] == self.label_queue[None, :]) & (labels[:, None] != -1)).float().cuda()
                targets_global = torch.cat([positive_target, targets], dim=1)
                targets_small = targets_global.repeat_interleave(repeats=self.args.num_crops[1], dim=0)
                # print("Target Global Min: {}, Max: {}".format(targets_global.min(), targets_global.max()))
                # print("Target Small Min: {}, Max: {}".format(targets_small.min(), targets_small.max()))
                labels_small = labels.repeat_interleave(repeats=self.args.num_crops[1], dim=0)

                # dequeue and enqueue
                if base_sess or (not base_sess and last_epochs_new):
                    self._dequeue_and_enqueue(k, labels)

                return logits_classify, logits_global, logits_small, targets_global, targets_small

        elif self.mode == 'encoder':
            x, _ = self.encode_q(im_cla)
            return x
        else:
            raise ValueError('Unknown mode')

    def update_fc(self, dataloader, class_list, session, transform=None):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            b = data.size()[0]
            data = transform(data)
            m = data.size()[0] // b
            labels = torch.stack([label * m + ii for ii in range(m)], 1).view(-1)
            data, _ = self.encode_q(data)
            data.detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list) * m, self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, labels, class_list, m=m)

    def update_fc_avg(self, data, labels, class_list, m=None):
        new_fc = []
        for class_index in class_list:
            for i in range(m):
                index = class_index * m + i
                data_index = (labels == index).nonzero().squeeze(-1)
                embedding = data[data_index]
                proto = embedding.mean(0)
                new_fc.append(proto)
                self.fc.weight.data[index] = proto
        new_fc = torch.stack(new_fc, dim=0)
        return new_fc

    def get_logits(self, x, fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x, fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))


# LIMIT (TPAMI 2023)
def sample_task_ids(support_label, num_task, num_shot, num_way, num_class):
    basis_matrix = torch.arange(num_shot).long().view(-1, 1).repeat(1, num_way).view(-1) * num_class
    permuted_ids = torch.zeros(num_task, num_shot * num_way).long()
    permuted_labels = []

    for i in range(num_task):
        clsmap = torch.randperm(num_class)[:num_way]
        permuted_labels.append(support_label[clsmap])
        permuted_ids[i, :].copy_(basis_matrix + clsmap.repeat(num_shot))

    return permuted_ids, permuted_labels


def one_hot(indices, depth):
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if indices.is_cuda:
        encoded_indicies = encoded_indicies.cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


class LIMITNet(BaseNet):
    def __init__(self, args, mode=None):
        super().__init__(args, mode)

        hdim = self.num_features
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        self.seminorm = True
        if args.dataset == 'cifar100':
            self.seminorm = False

    def split_instances(self, support_label, epoch):
        args = self.args
        # crriculum for num_way:
        total_epochs = args.epochs_base

        # Linear increment
        # self.current_way=int(5+float((args.sample_class-5)/total_epochs)*epoch)
        # Linear drop
        # self.current_way=int(args.sample_class-float((args.sample_class-5)/total_epochs)*epoch)
        # Equal
        # self.current_way=10
        # Random Sample
        # self.current_way=np.random.randint(5,args.sample_class)
        self.current_way = args.sample_class
        permuted_ids, permuted_labels = sample_task_ids(support_label, args.num_tasks, num_shot=args.sample_shot,
                                                        num_way=self.current_way, num_class=args.sample_class)
        index_label = (
        permuted_ids.view(args.num_tasks, args.sample_shot, self.current_way), torch.stack(permuted_labels))

        return index_label

    def forward(self, x_shot, x_query=None, shot_label=None, epoch=None):

        if self.mode == 'encoder':
            x_shot = self.encode(x_shot)
            return x_shot
        elif x_query is not None:
            support_emb = self.encode(x_shot)
            query_emb = self.encode(x_query)
            index_label = self.split_instances(shot_label, epoch)
            logits = self._forward(support_emb, query_emb, index_label)
            return logits
        elif self.mode != 'encoder':
            logits = self.forward_metric(x_shot)
            return logits

    def _forward(self, support, query, index_label):

        support_idx, support_labels = index_label
        num_task = support_idx.shape[0]
        num_dim = support.shape[-1]
        # organize support data
        support = support[support_idx.view(-1)].view(*(support_idx.shape + (-1,)))
        proto = support.mean(dim=1)  # Ntask x NK x d
        num_proto = proto.shape[1]
        logit = []

        num_batch = 1
        num_proto = self.args.num_classes
        num_query = query.shape[0]
        emb_dim = support.size(-1)
        query = query.unsqueeze(1)

        for tt in range(num_task):
            # combine proto with the global classifier
            global_mask = torch.eye(self.args.num_classes).cuda()
            whole_support_index = support_labels[tt, :]
            global_mask[:, whole_support_index] = 0
            # construct local mask
            local_mask = one_hot(whole_support_index, self.args.num_classes)
            current_classifier = torch.mm(self.fc.weight.t(), global_mask) + torch.mm(proto[tt, :].t(), local_mask)
            current_classifier = current_classifier.t()  # 100*64
            current_classifier = current_classifier.unsqueeze(0)

            current_classifier = current_classifier.unsqueeze(1).expand(num_batch, num_query, num_proto,
                                                                        emb_dim).contiguous()
            current_classifier = current_classifier.view(num_batch * num_query, num_proto, emb_dim)

            combined = torch.cat([current_classifier, query], 1)  # Nk x (N + 1) x d, batch_size = NK
            combined = self.slf_attn(combined, combined, combined)
            # compute distance for all batches
            current_classifier, query = combined.split(num_proto, 1)

            if self.seminorm:
                # norm classifier
                current_classifier = F.normalize(current_classifier, dim=-1)  # normalize for cosine distance
                logits = torch.bmm(query, current_classifier.permute([0, 2, 1])) / self.args.temperature
                logits = logits.view(-1, num_proto)
            else:
                # cosine
                logits = F.cosine_similarity(query, current_classifier, dim=-1)
                logits = logits * self.args.temperature

            logit.append(logits)
        logit = torch.cat(logit, 1)
        logit = logit.view(-1, self.args.num_classes)

        return logit

    def updateclf(self, data, label):
        support_embs = self.encode(data)
        num_dim = support_embs.shape[-1]
        # proto = support_embs.reshape(self.args.eval_shot, -1, num_dim).mean(dim=0) # N x d
        proto = support_embs.reshape(5, -1, num_dim).mean(dim=0)  # N x d
        cls_unseen, _, _ = self.slf_attn(proto.unsqueeze(0), self.shared_key, self.shared_key)
        # cls_unseen = F.normalize(cls_unseen.squeeze(0), dim=1)
        cls_unseen = cls_unseen.squeeze(0)
        self.fc.weight.data[torch.min(label):torch.max(label) + 1] = cls_unseen

    def forward_many(self, query):
        # cls_seen = F.normalize(self.fc.weight, dim=1)
        num_batch = 1
        num_proto = self.args.num_classes

        emb_dim = query.size(-1)
        query = query.view(-1, 1, emb_dim)
        num_query = query.shape[0]

        current_classifier = self.fc.weight.unsqueeze(0)
        current_classifier = current_classifier.unsqueeze(1).expand(num_batch, num_query, num_proto,
                                                                    emb_dim).contiguous()
        current_classifier = current_classifier.view(num_batch * num_query, num_proto, emb_dim)

        combined = torch.cat([current_classifier, query], 1)  # Nk x (N + 1) x d, batch_size = NK
        combined = self.slf_attn(combined, combined, combined)
        # compute distance for all batches
        current_classifier, query = combined.split(num_proto, 1)

        if self.seminorm:
            # norm classifier
            current_classifier = F.normalize(current_classifier, dim=-1)  # normalize for cosine distance
            logits = torch.bmm(query, current_classifier.permute([0, 2, 1])) / self.args.temperature
            logits = logits.view(-1, num_proto)
        else:
            # cosine
            logits = F.cosine_similarity(query, current_classifier, dim=-1)
            logits = logits * self.args.temperature
        return logits


class JointNet(BaseNet):
    def __init__(self, args, mode=None, use_norm=False):
        super().__init__(args, mode)

        self.mode = mode
        self.args = args

        if self.args.dataset in ['cifar100']:
            if self.args.etf:
                self.encoder = resnet20_etf()
            else:
                self.encoder = resnet20()
            self.num_features = 128
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet18(False, args)
            self.num_features = 512
        if self.args.dataset in ['cub200']:
            self.encoder = resnet18(True, args)
            self.num_features = 512

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if self.args.etf:
            self.fc = ETF_Classifier(self.num_features, self.args.num_classes)
        else:
            if use_norm:
                self.fc = NormedLinear(self.num_features, self.args.num_classes)
            else:
                self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
    
    def encode(self, x):
        x = self.encoder(x)
        
        if self.args.etf == False:
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)
        
        return x

    def forward_metric(self, x):
        x = self.encode(x)
        x = self.fc(x)
        
        return x

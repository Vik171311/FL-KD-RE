import copy

from torch import nn
from timm.models.layers import _assert, trunc_normal_
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

from feature_extraction import create_sub_network
from torchvision import models, transforms
import numpy as np
import torch.nn.functional as F
import torch
from cka_torch import cka_linear_torch, cka_rbf_torch
from lr_torch import lr_torch
from block_meta import *
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
class GetFeatureHook:

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.feature = []

    def hook_fn(self, module, input, output):
        if isinstance(input, tuple):
            input = input[0]
        self.in_size = input.shape[1:]
        self.out_size = output.shape[1:]
        output = F.adaptive_avg_pool2d(output, (1, 1)).detach().cpu()
        self.feature.append(output)

    def concat(self):
        self.feature = torch.cat(self.feature, dim=0)

    def flash_mem(self):
        del self.feature
        self.feature = []

    def close(self):
        self.hook.remove()

input_shape = (3, 224, 224)

def similarity_pair_batch_cka(data1, data2, bs=2048):
    feat1 = data1['feat']
    feat2 = data2['feat']
    num_layer1 = len(feat1.keys())
    num_layer2 = len(feat2.keys())
    name1 = data1['model_name']
    name2 = data2['model_name']
    print(f'number of layers in {name1} is {num_layer1}')
    print(f'number of layers in {name2} is {num_layer2}')
    num_layer1 = len(feat1.keys())
    num_layer2 = len(feat2.keys())

    num_sample = list(feat1.values())[0].shape[0]
    num_batch = np.int64(np.ceil(num_sample / bs))
    print(
        f'number of samples {num_sample}, number of batch {num_batch}, batch size {bs}')

    cka_map = torch.zeros((num_batch, num_layer1, num_layer2)).cuda()
    for b_id in range(num_batch):
        start = b_id*bs
        end = (b_id+1)*bs if (b_id+1)*bs < num_sample-1 else num_sample-1
        for i, (k1, v1) in enumerate(feat1.items()):
            for j, (k2, v2) in enumerate(feat2.items()):
                cka_from_examples = cka_linear_torch(
                    torch.tensor(v1[start:end]).cuda(),
                    torch.tensor(v2[start:end]).cuda())
                cka_map[b_id, i, j] = cka_from_examples
    return cka_map.mean(0).detach().cpu().numpy()

def similarity_pair_batch_rbf_cka(data1, data2, bs=2048):
    feat1 = data1['feat']
    feat2 = data2['feat']
    num_layer1 = len(feat1.keys())
    num_layer2 = len(feat2.keys())
    name1 = data1['model_name']
    name2 = data2['model_name']
    print(f'number of layers in {name1} is {num_layer1}')
    print(f'number of layers in {name2} is {num_layer2}')
    num_layer1 = len(feat1.keys())
    num_layer2 = len(feat2.keys())

    num_sample = list(feat1.values())[0].shape[0]
    num_batch = np.int64(np.ceil(num_sample / bs))
    print(
        f'number of samples {num_sample}, number of batch {num_batch}, batch size {bs}')

    cka_map = torch.zeros((num_batch, num_layer1, num_layer2)).cuda()
    for b_id in range(num_batch):
        start = b_id*bs
        end = (b_id+1)*bs if (b_id+1)*bs < num_sample-1 else num_sample-1
        for i, (k1, v1) in enumerate(feat1.items()):
            for j, (k2, v2) in enumerate(feat2.items()):
                cka_from_examples = cka_rbf_torch(
                    torch.tensor(v1[start:end]).cuda(),
                    torch.tensor(v2[start:end]).cuda())
                cka_map[b_id, i, j] = cka_from_examples
    return cka_map.mean(0).detach().cpu().numpy()

def similarity_pair_batch_lr(data1, data2, bs=2048):
    feat1 = data1['feat']
    feat2 = data2['feat']
    num_layer1 = len(feat1.keys())
    num_layer2 = len(feat2.keys())
    name1 = data1['model_name']
    name2 = data2['model_name']
    print(f'number of layers in {name1} is {num_layer1}')
    print(f'number of layers in {name2} is {num_layer2}')
    num_layer1 = len(feat1.keys())
    num_layer2 = len(feat2.keys())

    num_sample = list(feat1.values())[0].shape[0]
    num_batch = np.int64(np.ceil(num_sample / bs))
    print(
        f'number of samples {num_sample}, number of batch {num_batch}, batch size {bs}')

    cka_map = torch.zeros((num_batch, num_layer1, num_layer2)).cuda()
    for b_id in range(num_batch):
        start = b_id*bs
        end = (b_id+1)*bs if (b_id+1)*bs < num_sample-1 else num_sample-1
        for i, (k1, v1) in enumerate(feat1.items()):
            for j, (k2, v2) in enumerate(feat2.items()):
                cka_from_examples = lr_torch(
                    torch.tensor(v1[start:end]).cuda(),
                    torch.tensor(v2[start:end]).cuda())
                cka_map[b_id, i, j] = cka_from_examples
    return cka_map.mean(0).detach().cpu().numpy()

SIM_FUNC = {
    'cka':similarity_pair_batch_cka,
    'rbf_cka':similarity_pair_batch_rbf_cka,
    'lr':similarity_pair_batch_lr
}

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # if isinstance(obj, int64):
        return super(NpEncoder, self).default(obj)


def network_to_module_subnet(model_name, block_input, block_output, backend):
    backbone = getattr(models, model_name)(pretrained=True)
    if isinstance(block_input, str):
        block_input = [block_input]
    elif isinstance(block_input, tuple):
        block_input = list(block_input)
    elif isinstance(block_input, list):
        block_input = block_input


    if isinstance(block_output, str):
        block_output = [block_output]
    elif isinstance(block_output, tuple):
        block_output = list(block_output)
    elif isinstance(block_output, list):
        block_output = block_output
    else:
        TypeError('Block output should be a string or tuple or list')


    subnet = create_sub_network(backbone, block_input, block_output)
    return subnet


class Block:
    def __init__(self, model_name, block_index, node_list):
        assert isinstance(model_name, str)
        assert isinstance(node_list, list)
        for i in range(len(node_list)-1):
            assert node_list[i+1] - node_list[i] == 1, node_list
        self.model_name = model_name
        self.block_index = block_index
        self.node_list = node_list
        # print(model_name)
        self.value = 0  # MODEL_STATS[self.model_name]['top1']
        self.size = 0
        self.group_id = None

    def print_split(self):
        start = self.node_list[0]
        end = self.node_list[-1]
        return [MODEL_STATS[self.model_name]['arch'],
                MODEL_BLOCKS[self.model_name][start],
                MODEL_BLOCKS[self.model_name][end],
                MODEL_STATS[self.model_name]['backend']]

    def get_inout_size(self,MODEL_INOUT_SHAPE):
        start = self.node_list[0]
        end = self.node_list[-1]
        start_name = MODEL_BLOCKS[self.model_name][start]
        end_name = MODEL_BLOCKS[self.model_name][end]
        self.in_size = MODEL_INOUT_SHAPE[self.model_name]['in_size'][start_name]
        self.out_size = MODEL_INOUT_SHAPE[self.model_name]['out_size'][end_name]

    def __len__(self):
        return len(self.node_list)

    def get_model_size(self):
        model, block_input, block_output, backend = self.print_split()
        block = network_to_module_subnet(
            model, block_input, block_output, backend)
        self.size = sum(p.numel() for p in block.parameters())/1e6

    def __eq__(self, other):
        if isinstance(other, Block):
            return (self.model_name == other.model_name and
                    self.block_index == other.block_index and
                    self.node_list == other.node_list)
        else:
            return False

    def __repr__(self) -> str:
        return json.dumps(self.__dict__, cls=NpEncoder, indent=2)

    def __str__(self):
        nodes_in = str(self.node_list[0])
        node_out = str(self.node_list[-1])
        return f'{MODEL_PRINT[self.model_name]}:{nodes_in}-{node_out} Stage-{self.block_index}'
        # return f'Model Name:{self.model_name}\tNode list:{self.node_list}\tBlock Index: {self.block_index}\t Size:{self.size}'


class Block_Sim:
    def __init__(self, sim_dict):
        self.sim_dict = sim_dict

    def get_sim(self, block1, block2):
        if isinstance(block1, Block) and isinstance(block2, Block):
            # print(block1, block2)
            key = f'{block1.model_name}.{block2.model_name}'
            if key in self.sim_dict.keys():
                if block1 == block2:
                    block_sim = 1
                elif block1.model_name == block1.model_name and block1.block_index != block2.block_index:
                    block_sim = 0
                else:
                    sim_map = self.sim_dict[key]
                    try:
                        block_sim = (sim_map[block1.node_list[0], block2.node_list[0]] +
                                     sim_map[block1.node_list[-1], block2.node_list[-1]])
                    except:
                        AssertionError('The functional similarity can not be computed')
            else:
                block_sim = 0

            return block_sim
        else:
            TypeError('block 1 and block 2 must be Block instance')


class Block_Assign:
    def __init__(self, assignment_index, block_split_dict, centers):
        self.block2center = dict()
        self.center2block = [[c]for c in centers]
        self.centers = centers

        for m, model_name in enumerate(MODEL_ZOO):
            self.block2center[model_name] = dict()
            for j, block in enumerate(block_split_dict[model_name]):
                center_index = assignment_index[m, j]
                block.group_id = center_index
                self.block2center[model_name][j] = centers[center_index]
                self.center2block[center_index].append(block)

    def get_center(self, block):
        return self.block2center[block.model_name][block.block_index]

    def print_assignment(self):
        results = ''
        for i, group in enumerate(self.center2block):
            results += 'Center {}\n'.format(str(self.centers[i]))
            results += '\n'.join(['\t'+str(c) for c in group])
            results += '\n'
        print(results)

    def get_size(self,MODEL_INOUT_SHAPE):
        for group in self.center2block:
            for block in group:
                # block.get_model_size()
                block.get_inout_size(MODEL_INOUT_SHAPE)
                print(block)



def check_valid(selected_block):
    cnn_max = 0
    vit_min = len(selected_block)
    for s in selected_block:
        if s is not None:
            if (s.model_name.startswith('vit') or s.model_name.startswith('swin')):
                if s.block_index < vit_min:
                    vit_min = s.block_index
            else:
                if s.block_index > cnn_max:
                    cnn_max = s.block_index

    return cnn_max < vit_min


class GetFeatureHook3340857:

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.feature = []

    def hook_fn(self, module, input, output):
        if isinstance(input, tuple):
            input = input[0]
        self.in_size = input.shape[1:]
        self.out_size = output.shape[1:]
        output = F.adaptive_avg_pool2d(output, (1, 1)).detach().cpu()
        self.feature.append(output)

    def concat(self):
        self.feature = torch.cat(self.feature, dim=0)

    def flash_mem(self):
        del self.feature
        self.feature = []

    def close(self):
        self.hook.remove()


class ConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, targets=None, global_protos=None, mask=None):
        """Compute contrastive loss between feature and global prototype
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if targets is not None and mask is not None:
            raise ValueError('Cannot define both `targets` and `mask`')
        elif targets is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif targets is not None:
            targets = targets.contiguous().view(-1, 1)
            if targets.shape[0] != batch_size:
                raise ValueError('Num of targets does not match num of features')
            mask = torch.eq(targets, targets.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            # anchor_feature = contrast_feature
            anchor_count = contrast_count
            anchor_feature = torch.zeros_like(contrast_feature)
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # generate anchor_feature
        for i in range(batch_size*anchor_count):
            anchor_feature[i, :] = global_protos[targets[i%batch_size].item()]

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def agg_func(protos):

    agg_protos = {}
    for [label, proto_list] in protos.items():
        proto = torch.stack(proto_list)
        agg_protos[label] = torch.mean(proto, dim=0).data

    return agg_protos

def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def average_weights(w):

    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        for i in range(1, len(w)):
            w_avg[0][key] += w[i][key]
        w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
        for i in range(1, len(w)):
            w_avg[i][key] = w_avg[0][key]
    return w_avg

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm, act_layer=nn.Identity):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)

        in_features = 4 * dim
        self.reduction = nn.Linear(in_features, self.out_dim, bias=False)
        self.act = act_layer()

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        _assert(L == H * W, "input feature has wrong size")
        _assert(H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even.")

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        x = self.act(x)

        return x


class GAP1d(nn.Module):
    def __init__(self):
        super(GAP1d, self).__init__()

    def forward(self, x):
        return x.mean(1)


class TokenFilter(nn.Module):
    """remove cls tokens in forward"""

    def __init__(self, number=1, inverse=False, remove_mode=True):
        super(TokenFilter, self).__init__()
        self.number = number
        self.inverse = inverse
        self.remove_mode = remove_mode

    def forward(self, x):
        if self.inverse and self.remove_mode:
            x = x[:, :-self.number, :]
        elif self.inverse and not self.remove_mode:
            x = x[:, -self.number:, :]
        elif not self.inverse and self.remove_mode:
            x = x[:, self.number:, :]
        else:
            x = x[:, :self.number, :]
        return x


class TokenFnContext(nn.Module):
    def __init__(self, token_num=0, fn: nn.Module = nn.Identity(), token_fn: nn.Module = nn.Identity(), inverse=False):
        super(TokenFnContext, self).__init__()
        self.token_num = token_num
        self.fn = fn
        self.token_fn = token_fn
        self.inverse = inverse
        self.token_filter = TokenFilter(number=token_num, inverse=inverse, remove_mode=False)
        self.feature_filter = TokenFilter(number=token_num, inverse=inverse)

    def forward(self, x):
        tokens = self.token_filter(x)
        features = self.feature_filter(x)
        features = self.fn(features)
        if self.token_num == 0:
            return features

        tokens = self.token_fn(tokens)
        if self.inverse:
            x = torch.cat([features, tokens], dim=1)
        else:
            x = torch.cat([tokens, features], dim=1)
        return x


class LambdaModule(nn.Module):
    def __init__(self, lambda_fn):
        super(LambdaModule, self).__init__()
        self.fn = lambda_fn

    def forward(self, x):
        return self.fn(x)


class MyPatchMerging(nn.Module):
    def __init__(self, out_patch_num):
        super().__init__()
        self.out_patch_num = out_patch_num

    def forward(self, x):
        B, L, D = x.shape
        patch_size = int(L ** 0.5)
        assert patch_size ** 2 == L
        out_patch_size = int(self.out_patch_num ** 0.5)
        assert out_patch_size ** 2 == self.out_patch_num
        grid_size = patch_size // out_patch_size
        assert grid_size * out_patch_size == patch_size
        x = x.view(B, out_patch_size, grid_size, out_patch_size, grid_size, D)
        x = torch.einsum('bhpwqd->bhwpqd', x)
        x = x.reshape(shape=(B, out_patch_size ** 2, -1))
        return x


class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        #   depthwise and pointwise convolution, downsample by 2
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in,
                      bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


def init_weights(module):
    for n, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def kd_loss(logits_student, logits_teacher, temperature=1.):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
    loss_kd *= temperature ** 2
    return loss_kd


def is_cnn_model(distiller):
    if hasattr(distiller, 'module'):
        _, sizes = distiller.module.stage_info(1)
    else:
        _, sizes = distiller.stage_info(1)
    if len(sizes) == 3:  # C H W
        return True
    elif len(sizes) == 2:  # L D
        return False
    else:
        raise RuntimeError('unknown model feature shape')


def set_module_dict(module_dict, k, v):
    if not isinstance(k, str):
        k = str(k)
    module_dict[k] = v


def get_module_dict(module_dict, k):
    if not isinstance(k, str):
        k = str(k)
    return module_dict[k]


def patchify(imgs, p):
    """
    imgs: (N, C, H, W)
    x: (N, L, patch_size**2, C)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    in_chans = imgs.shape[1]
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], in_chans, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * in_chans))
    return x


class Unpatchify(nn.Module):
    def __init__(self, p):
        super(Unpatchify, self).__init__()
        self.p = p

    def forward(self, x):
        return _unpatchify(x, self.p)


def _unpatchify(x, p):
    """
    x: (N, L, patch_size**2 *C)
    imgs: (N, C, H, W)
    """
    # p = self.patch_embed.patch_size[0]
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
    return imgs


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    return net_cls_counts

def load_data(args):
    np.random.seed(10)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset=="CIFAR10":
        train_dataset=CIFAR10('data',train=True,transform=transform_train)
        test_dataset=CIFAR10('data',train=False,transform=transform_test)
    elif args.dataset=="CIFAR100":
        train_dataset=CIFAR100('data',train=True,transform=transform_train)
        test_dataset=CIFAR100('data',train=False,transform=transform_test)
    elif args.dataset=="SVHN":
        train_dataset=SVHN('data',split='train',download=True,transform=transform_train)
        test_dataset=SVHN('data',split='test',download=True,transform=transform_test)
    train_dataset.targets=np.array(train_dataset.labels)
    return  train_dataset,test_dataset


def split_data(args,train_dataset,test_dataset):
    n_train = train_dataset.targets.shape[0]
    if args.partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, args.clientnum)
        net_dataidx_map = {i: batch_idxs[i] for i in range(args.clientnum)}

    elif args.partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        if args.dataset == 'cifar100':
            K = 100
        elif args.dataset == 'tinyimagenet':
            K = 200
            # min_require_size = 100

        N = train_dataset.targets.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(args.clientnum)]
            for k in range(K):
                idx_k = np.where(train_dataset.targets == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(args.beta, args.clientnum))
                proportions = np.array([p * (len(idx_j) < N / args.clientnum) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(args.clientnum):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]



        # net_dataidx_map=


    traindata_cls_counts = record_net_data_stats(train_dataset.targets, net_dataidx_map)
    return net_dataidx_map,traindata_cls_counts
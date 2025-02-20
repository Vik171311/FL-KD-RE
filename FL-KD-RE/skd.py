import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(10)
random.seed(10)
torch.cuda.manual_seed(10)
torch.cuda.manual_seed_all(10)
np.random.seed(10)
from timm.models.vision_transformer import Block
from utils import GAP1d, get_module_dict, init_weights, is_cnn_model, PatchMerging, SepConv, set_module_dict, \
    TokenFilter, TokenFnContext

class BaseDistiller(nn.Module):
    def __init__(self, student, teacher1,teacher2, teacher3,teacher4,criterion, args):
        super(BaseDistiller, self).__init__()
        self.student = student
        self.teacher1 = teacher1
        self.teacher2 = teacher2
        self.teacher3 = teacher3
        self.teacher4 = teacher4
        self.criterion = criterion
        self.args = args

    def forward(self, image, label, *args, **kwargs):
        raise NotImplementedError

    def get_learnable_parameters(self):
        student_params = 0
        extra_params = 0
        for n, p in self.named_parameters():
            if n.startswith('student'):
                student_params += p.numel()
            elif n.startswith('teacher'):
                continue
            else:
                if p.requires_grad:
                    extra_params += p.numel()
        return student_params, extra_params


# class Vanilla(BaseDistiller):
#     requires_feat = False
#
#     def __init__(self, student, teacher, criterion, args, **kwargs):
#         super(Vanilla, self).__init__(student, teacher, criterion, args)
#
#     def forward(self, image, label, *args, **kwargs):
#         logits_student = self.student(image)
#
#         loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
#         losses_dict = {
#             "loss_gt": loss_gt,
#         }
#         return logits_student, losses_dict
def skd_loss(logits_student, logits_teacher1, logits_teacher2,logits_teacher3, logits_teacher4, target_mask, eps, args,temperature=1.):
    pred_student = F.softmax(logits_student / temperature, dim=1)

    pred_teacher1 = F.softmax(logits_teacher1 / temperature, dim=1)
    pred_teacher2 = F.softmax(logits_teacher2 / temperature, dim=1)
    pred_teacher3 = F.softmax(logits_teacher3 / temperature, dim=1)
    pred_teacher4 = F.softmax(logits_teacher4 / temperature, dim=1)

    prod = (args.a * pred_teacher1 + args.b * pred_teacher2 + args.c * pred_teacher3 + args.d * pred_teacher4 + target_mask) ** eps
    loss = torch.sum(- (prod - target_mask) * torch.log(pred_student), dim=-1)
    return loss.mean()

def skd_loss2(logits_student, logits_teacher1, logits_teacher2,target_mask, eps, args,temperature=1.):
    pred_student = F.softmax(logits_student / temperature, dim=1)

    pred_teacher1 = F.softmax(logits_teacher1 / temperature, dim=1)
    pred_teacher2 = F.softmax(logits_teacher2 / temperature, dim=1)


    prod = (args.A * pred_teacher1 + args.B * pred_teacher2 + target_mask) ** eps
    loss = torch.sum(- (prod - target_mask) * torch.log(pred_student), dim=-1)
    return loss.mean()


class SKD(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher1,teacher2,teacher3,teacher4, criterion, args,best_block_split_dict, **kwargs):
        super(SKD, self).__init__(student, teacher1, teacher2,teacher3,teacher4, criterion, args)
        self.args=args
        self.best_block_split_dict = best_block_split_dict
        if len(self.args.skd_eps) == 1:
            eps = [self.args.skd_eps[0] for _ in range(args.K + 1)]
            self.args.skd_eps = eps

        assert args.K + 1 == len(self.args.skd_eps)

        self.projector_teacher = nn.ModuleDict().cuda()
        self.projector_teacher2 = nn.ModuleDict().cuda()
        self.projector_teacher3 = nn.ModuleDict().cuda()
        self.projector_teacher4 = nn.ModuleDict().cuda()
        self.projector_student = nn.ModuleDict().cuda()

        is_cnn_student = is_cnn_model(student)

        _, feature_dim_t = self.teacher1.stage_info(-1)
        _, feature_dim_t2 = self.teacher2.stage_info(-1)
        _, feature_dim_t3 = self.teacher3.stage_info(-1)
        _, feature_dim_t4 = self.teacher4.stage_info(-1)
        _, feature_dim_s = self.student.stage_info(-1)

        self.student_index=[]
        self.teacher_index=[]
        self.teacher_index2=[]
        self.teacher_index3 = []
        self.teacher_index4 = []
    def get_teacher_out(self,model):
        _, feature_dim_t = self.teacher1.stage_info(-1)
        _, feature_dim_s = self.student.stage_info(-1)
        for stage in range(self.args.K):
            self.teacher_index.append(self.best_block_split_dict[model][stage].node_list[-1])
            _, size_s = self.teacher1.stage_info(self.best_block_split_dict[model][stage].node_list[-1])
            in_chans, _, _ = size_s
            if stage != self.args.K - 1:
                down_sample_blks = []
                while in_chans < max(feature_dim_s, feature_dim_t):
                    out_chans = in_chans * 2
                    down_sample_blks.append(SepConv(in_chans, out_chans))
                    in_chans *= 2
            else:
                down_sample_blks = [nn.Conv2d(in_chans, max(feature_dim_s, feature_dim_t), 1, 1, 0)]

            projector = nn.Sequential(
                *down_sample_blks,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(max(feature_dim_s, feature_dim_t), self.args.num_classes)  # todo: cifar100
            ).cuda()

            set_module_dict(self.projector_teacher, stage, projector)
        self.projector_teacher.apply(init_weights)
        return self.projector_teacher

    def get_teacher_out2(self,model):
        _, feature_dim_t = self.teacher2.stage_info(-1)
        _, feature_dim_s = self.student.stage_info(-1)
        for stage in range(self.args.K):
            self.teacher_index2.append(self.best_block_split_dict[model][stage].node_list[-1])
            _, size_s = self.teacher2.stage_info(self.best_block_split_dict[model][stage].node_list[-1])
            in_chans, _, _ = size_s
            if stage != self.args.K - 1:
                down_sample_blks = []
                while in_chans < max(feature_dim_s, feature_dim_t):
                    out_chans = in_chans * 2
                    down_sample_blks.append(SepConv(in_chans, out_chans))
                    in_chans *= 2
            else:
                down_sample_blks = [nn.Conv2d(in_chans, max(feature_dim_s, feature_dim_t), 1, 1, 0)]

            projector = nn.Sequential(
                *down_sample_blks,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(max(feature_dim_s, feature_dim_t), self.args.num_classes)  # todo: cifar100
            ).cuda()

            set_module_dict(self.projector_teacher2, stage, projector)
        self.projector_teacher2.apply(init_weights)
        return self.projector_teacher2

    def get_teacher_out3(self,model):
        _, feature_dim_t = self.teacher3.stage_info(-1)
        _, feature_dim_s = self.student.stage_info(-1)
        for stage in range(self.args.K):
            self.teacher_index3.append(self.best_block_split_dict[model][stage].node_list[-1])
            _, size_s = self.teacher3.stage_info(self.best_block_split_dict[model][stage].node_list[-1])
            in_chans, _, _ = size_s
            if stage != self.args.K - 1:
                down_sample_blks = []
                while in_chans < max(feature_dim_s, feature_dim_t):
                    out_chans = in_chans * 2
                    down_sample_blks.append(SepConv(in_chans, out_chans))
                    in_chans *= 2
            else:
                down_sample_blks = [nn.Conv2d(in_chans, max(feature_dim_s, feature_dim_t), 1, 1, 0)]

            projector = nn.Sequential(
                *down_sample_blks,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(max(feature_dim_s, feature_dim_t), self.args.num_classes)  # todo: cifar100
            ).cuda()

            set_module_dict(self.projector_teacher3, stage, projector)
        self.projector_teacher3.apply(init_weights)
        return self.projector_teacher3


    def get_teacher_out4(self,model):
        _, feature_dim_t = self.teacher4.stage_info(-1)
        _, feature_dim_s = self.student.stage_info(-1)
        for stage in range(self.args.K):
            self.teacher_index4.append(self.best_block_split_dict[model][stage].node_list[-1])
            _, size_s = self.teacher4.stage_info(self.best_block_split_dict[model][stage].node_list[-1])
            in_chans, _, _ = size_s
            if stage != self.args.K - 1:
                down_sample_blks = []
                while in_chans < max(feature_dim_s, feature_dim_t):
                    out_chans = in_chans * 2
                    down_sample_blks.append(SepConv(in_chans, out_chans))
                    in_chans *= 2
            else:
                down_sample_blks = [nn.Conv2d(in_chans, max(feature_dim_s, feature_dim_t), 1, 1, 0)]

            projector = nn.Sequential(
                *down_sample_blks,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(max(feature_dim_s, feature_dim_t), self.args.num_classes)  # todo: cifar100
            ).cuda()

            set_module_dict(self.projector_teacher4, stage, projector)
        self.projector_teacher4.apply(init_weights)
        return self.projector_teacher4
    def get_student_out(self,model):
        _, feature_dim_t = self.teacher1.stage_info(-1)
        _, feature_dim_s = self.student.stage_info(-1)
        for stage in range(self.args.K):
            self.student_index.append(self.best_block_split_dict[model][stage].node_list[-1])
            _, size_s = self.student.stage_info(self.best_block_split_dict[model][stage].node_list[-1])
            in_chans, _, _ = size_s
            if stage != self.args.K - 1:
                down_sample_blks = []
                while in_chans < max(feature_dim_s, feature_dim_t):
                    out_chans = in_chans * 2
                    down_sample_blks.append(SepConv(in_chans, out_chans))
                    in_chans *= 2
            else:
                down_sample_blks = [nn.Conv2d(in_chans, max(feature_dim_s, feature_dim_t), 1, 1, 0)]

            projector = nn.Sequential(
                *down_sample_blks,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(max(feature_dim_s, feature_dim_t), self.args.num_classes)  # todo: cifar100
            ).cuda()

            set_module_dict(self.projector_student, stage, projector)
        self.projector_student.apply(init_weights)
        return self.projector_student



    def forward(self, image, label, *args, **kwargs):
        self.projector_teacher = self.get_teacher_out('resnet34')
        self.projector_teacher2 = self.get_teacher_out2('resnet34_2')
        self.projector_teacher3 = self.get_teacher_out3('resnet34_3')
        self.projector_teacher4 = self.get_teacher_out4('resnet34_4')
        self.projector_student = self.get_student_out('resnet18')
        with torch.no_grad():
            self.teacher1.eval()
            logits_teacher,feat_teacher = self.teacher1(image,requires_feat=True)
            self.teacher2.eval()
            logits_teacher2, feat_teacher2 = self.teacher2(image, requires_feat=True)
            self.teacher3.eval()
            logits_teacher3, feat_teacher3 = self.teacher3(image, requires_feat=True)
            self.teacher4.eval()
            logits_teacher4, feat_teacher4 = self.teacher4(image, requires_feat=True)

        logits_student, feat_student = self.student(image, requires_feat=True)


        num_classes = logits_student.size(-1)
        if len(label.shape) != 1:  # label smoothing
            target_mask = F.one_hot(label.argmax(-1), num_classes)
        else:
            target_mask = F.one_hot(label, num_classes)

        skd_losses = []
        for stage,stage_student,stage_teacher1,stage_teacher2,stage_teacher3,stage_teacher4, eps in zip(range(self.args.K),self.student_index, self.teacher_index,self.teacher_index2, self.teacher_index3,self.teacher_index4,self.args.skd_eps):
            idx_s, _ = self.student.stage_info(stage_student)
            feat_s = feat_student[idx_s]
            logits_student_head = get_module_dict(self.projector_student, stage)(feat_s)

            idx_t, _ = self.teacher1.stage_info(stage_teacher1)
            feat_t = feat_teacher[idx_t]
            logits_teacher_head1 = get_module_dict(self.projector_teacher, stage)(feat_t)

            idx_t2, _ = self.teacher2.stage_info(stage_teacher2)
            feat_t2 = feat_teacher2[idx_t2]
            logits_teacher_head2 = get_module_dict(self.projector_teacher2, stage)(feat_t2)

            idx_t3, _ = self.teacher3.stage_info(stage_teacher3)
            feat_t3 = feat_teacher3[idx_t3]
            logits_teacher_head3 = get_module_dict(self.projector_teacher3, stage)(feat_t3)

            idx_t4, _ = self.teacher4.stage_info(stage_teacher4)
            feat_t4 = feat_teacher4[idx_t4]
            logits_teacher_head4 = get_module_dict(self.projector_teacher4, stage)(feat_t4)

            # skd_losses.append(
            #     skd_loss(logits_student_head, logits_teacher, target_mask, eps, self.args.skd_temperature))
            skd_losses.append(
                skd_loss(logits_student_head, logits_teacher_head1,logits_teacher_head2,logits_teacher_head3,logits_teacher_head4, target_mask, eps, self.args,self.args.skd_temperature))

        loss_skd = self.args.skd_loss_weight * sum(skd_losses)

        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        # loss_kd = self.args.kd_loss_weight * skd_loss2(logits_student, logits_teacher,logits_teacher2, target_mask,
        #                                               self.args.skd_eps[-1],self.args, self.args.skd_temperature)
        losses_dict = {
            "loss_gt": loss_gt,
            # "loss_kd": loss_kd,
            "loss_skd": loss_skd
        }
        return logits_student, losses_dict

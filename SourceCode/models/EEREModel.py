import torch

from SourceCode.utils.constant import pos_dict, CUDA

torch.manual_seed(1741)
import random
random.seed(1741)
import numpy as np
np.random.seed(1741)
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
import os.path as path
import os

class EERRoberta(nn.Module):
    def __init__(self, roberta_type, datasets,
                finetune=True, pos_dim=None, sub=True, mul=True, fn_activate='relu',
                negative_slope=0.2, drop_rate=0.5, task_weights=None, temperature=None):
        super().__init__()

        if path.exists("./pretrained_models/{}".format(roberta_type)):
            print("Loading pretrain model from local ......")
            self.roberta = RobertaModel.from_pretrained("./pretrained_models/{}".format(roberta_type))
        else:
            print("Loading pretrain model ......")
            self.roberta = RobertaModel.from_pretrained(roberta_type)

        if roberta_type == 'roberta-base':
            self.roberta_dim = 768
        if roberta_type == 'roberta-large':
            self.roberta_dim = 1024

        self.sub = sub
        self.mul = mul
        self.finetune = finetune

        if pos_dim != None:
            self.is_pos_emb = True
            pos_size = len(pos_dict.keys())
            self.pos_emb = nn.Embedding(pos_size, pos_dim)
            self.mlp_in = self.roberta_dim + pos_dim
        else:
            self.is_pos_emb = False
            self.mlp_in = self.roberta_dim

        self.mlp_size = self.roberta_dim

        self.drop_out = nn.Dropout(drop_rate)

        if fn_activate=='relu':
            self.relu = nn.LeakyReLU(negative_slope, True)
        elif fn_activate=='tanh':
            self.relu = nn.Tanh()
        elif fn_activate=='relu6':
            self.relu = nn.ReLU6()
        elif fn_activate=='silu':
            self.relu = nn.SiLU()
        elif fn_activate=='hardtanh':
            self.relu = nn.Hardtanh()

        self.max_num_class = 0

        module_dict = {}
        loss_dict = {}
        for dataset in datasets:
            if dataset == "HiEve":
                num_classes = 4
                if self.max_num_class < num_classes:
                    self.max_num_class = num_classes

                if sub==True and mul==True:
                    fc1 = nn.Linear(self.mlp_in*4, int(self.mlp_size*2))
                    fc2 = nn.Linear(int(self.mlp_size*2), num_classes)

                    h1 = nn.Linear(self.mlp_in*4, int(self.mlp_size*2))
                    h2 = nn.Linear(self.mlp_size * 2, int(self.mlp_size))

                if (sub==True and mul==False) or (sub==False and mul==True):
                    fc1 = nn.Linear(self.mlp_in*3, int(self.mlp_size*1.5))
                    fc2 = nn.Linear(int(self.mlp_size*1.5), num_classes)

                    h1 = nn.Linear(self.mlp_in*3, int(self.mlp_size*1.5))
                    h2 = nn.Linear(self.mlp_size * 1.5, int(self.mlp_size))

                if sub==False and mul==False:
                    fc1 = nn.Linear(self.mlp_in*2, int(self.mlp_size))
                    fc2 = nn.Linear(int(self.mlp_size), num_classes)

                    h1 = nn.Linear(self.mlp_in*2, int(self.mlp_size))
                    h2 = nn.Linear(self.mlp_size, int(self.mlp_size) // 2)

                # weights = [993.0/333, 993.0/349, 933.0/128, 933.0/453]
                # weights = torch.tensor(weights)
                # loss = nn.CrossEntropyLoss(weight=weights)

                module_dict['1'] = nn.Sequential(OrderedDict([
                                                ('dropout1',self.drop_out),
                                                ('fc1', fc1),
                                                ('dropout2', self.drop_out),
                                                ('relu', self.relu),
                                                ('fc2',fc2) ]))

                self.module_head = nn.Sequential(OrderedDict([
                                                ('h1', h1),
                                                ('relu', self.relu),
                                                ('h2', h2)]))

                # loss_dict['1'] = loss

            if dataset == "MATRES":
                num_classes = 4
                if self.max_num_class < num_classes:
                    self.max_num_class = num_classes

                if sub==True and mul==True:
                    fc1 = nn.Linear(self.mlp_in*4, int(self.mlp_size*2))
                    fc2 = nn.Linear(int(self.mlp_size*2), num_classes)

                    h1 = nn.Linear(self.mlp_in*4, int(self.mlp_size*2))
                    h2 = nn.Linear(self.mlp_size * 2, int(self.mlp_size))

                if (sub==True and mul==False) or (sub==False and mul==True):
                    fc1 = nn.Linear(self.mlp_in*3, int(self.mlp_size*1.5))
                    fc2 = nn.Linear(int(self.mlp_size*1.5), num_classes)

                    h1 = nn.Linear(self.mlp_in*3, int(self.mlp_size*1.5))
                    h2 = nn.Linear(self.mlp_size * 1.5, int(self.mlp_size))

                if sub==False and mul==False:
                    fc1 = nn.Linear(self.mlp_in*2, int(self.mlp_size))
                    fc2 = nn.Linear(int(self.mlp_size), num_classes)

                    h1 = nn.Linear(self.mlp_in*2, int(self.mlp_size))
                    h2 = nn.Linear(self.mlp_size, int(self.mlp_size) // 2)

                # weights = [30.0/412, 30.0/263, 30.0/30, 30.0/113,]
                #                 #                 # weights = torch.tensor(weights)
                #                 #                 # loss = nn.CrossEntropyLoss(weight=weights)

                module_dict['2'] = nn.Sequential(OrderedDict([
                                                ('dropout1',self.drop_out),
                                                ('fc1', fc1),
                                                ('dropout2', self.drop_out),
                                                ('relu', self.relu),
                                                ('fc2',fc2)]))
                # loss_dict['2'] = loss

                self.module_head = nn.Sequential(OrderedDict([
                                                ('h1', h1),
                                                ('relu', self.relu),
                                                ('h2', h2)]))
                # self.loss_sc = SupConLoss(temperature)

            if dataset == "I2B2":
                num_classes = 3
                if self.max_num_class < num_classes:
                    self.max_num_class = num_classes

                if sub==True and mul==True:
                    fc1 = nn.Linear(self.mlp_in*4, int(self.mlp_size*2))
                    fc2 = nn.Linear(int(self.mlp_size*2), num_classes)

                    h1 = nn.Linear(self.mlp_in*4, int(self.mlp_size*2))
                    h2 = nn.Linear(self.mlp_size * 2, int(self.mlp_size))

                if (sub==True and mul==False) or (sub==False and mul==True):
                    fc1 = nn.Linear(self.mlp_in*3, int(self.mlp_size*1.5))
                    fc2 = nn.Linear(int(self.mlp_size*1.5), num_classes)

                    h1 = nn.Linear(self.mlp_in*3, int(self.mlp_size*1.5))
                    h2 = nn.Linear(self.mlp_size * 1.5, int(self.mlp_size))

                if sub==False and mul==False:
                    fc1 = nn.Linear(self.mlp_in*2, int(self.mlp_size))
                    fc2 = nn.Linear(int(self.mlp_size), num_classes)

                    h1 = nn.Linear(self.mlp_in*2, int(self.mlp_size))
                    h2 = nn.Linear(self.mlp_size, int(self.mlp_size) // 2)

                # weights = [213.0/368, 213.0/213, 213.0/1013]
                # weights = torch.tensor(weights)
                # loss = nn.CrossEntropyLoss(weight=weights)

                module_dict['3'] = nn.Sequential(OrderedDict([
                                                ('dropout1',self.drop_out),
                                                ('fc1', fc1),
                                                ('dropout2', self.drop_out),
                                                ('relu', self.relu),
                                                ('fc2',fc2) ]))

                self.module_head = nn.Sequential(OrderedDict([
                                                ('h1', h1),
                                                ('relu', self.relu),
                                                ('h2', h2)]))
                # loss_dict['3'] = loss

            if dataset == "TBD":
                num_classes = 6
                if self.max_num_class < num_classes:
                    self.max_num_class = num_classes

                if sub==True and mul==True:
                    fc1 = nn.Linear(self.mlp_in*4, int(self.mlp_size*2))
                    fc2 = nn.Linear(int(self.mlp_size*2), num_classes)

                    h1 = nn.Linear(self.mlp_in*4, int(self.mlp_size*2))
                    h2 = nn.Linear(self.mlp_size * 2, int(self.mlp_size))

                if (sub==True and mul==False) or (sub==False and mul==True):
                    fc1 = nn.Linear(self.mlp_in*3, int(self.mlp_size*1.5))
                    fc2 = nn.Linear(int(self.mlp_size*1.5), num_classes)

                    h1 = nn.Linear(self.mlp_in*3, int(self.mlp_size*1.5))
                    h2 = nn.Linear(self.mlp_size * 1.5, int(self.mlp_size))

                if sub==False and mul==False:
                    fc1 = nn.Linear(self.mlp_in*2, int(self.mlp_size))
                    fc2 = nn.Linear(int(self.mlp_size), num_classes)

                    h1 = nn.Linear(self.mlp_in*2, int(self.mlp_size))
                    h2 = nn.Linear(self.mlp_size, int(self.mlp_size) // 2)

                # weights = [41.0/387, 41.0/287, 41.0/64, 41.0/74, 41.0/41, 41.0/642]
                # weights = torch.tensor(weights)
                # loss = nn.CrossEntropyLoss(weight=weights)

                module_dict['4'] = nn.Sequential(OrderedDict([
                                                ('dropout1',self.drop_out),
                                                ('fc1', fc1),
                                                ('dropout2', self.drop_out),
                                                ('relu', self.relu),
                                                ('fc2',fc2) ]))

                self.module_head = nn.Sequential(OrderedDict([
                                                ('h1', h1),
                                                ('relu', self.relu),
                                                ('h2', h2)]))

                # loss_dict['4'] = loss

            if "TDD" in dataset:
                num_classes = 5
                if self.max_num_class < num_classes:
                    self.max_num_class = num_classes

                if sub==True and mul==True:
                    fc1 = nn.Linear(self.mlp_in*4, int(self.mlp_size*2))
                    fc2 = nn.Linear(int(self.mlp_size*2), num_classes)

                    h1 = nn.Linear(self.mlp_in*4, int(self.mlp_size*2))
                    h2 = nn.Linear(self.mlp_size * 2, int(self.mlp_size))

                if (sub==True and mul==False) or (sub==False and mul==True):
                    fc1 = nn.Linear(self.mlp_in*3, int(self.mlp_size*1.5))
                    fc2 = nn.Linear(int(self.mlp_size*1.5), num_classes)

                    h1 = nn.Linear(self.mlp_in*3, int(self.mlp_size*1.5))
                    h2 = nn.Linear(self.mlp_size * 1.5, int(self.mlp_size))

                if sub==False and mul==False:
                    fc1 = nn.Linear(self.mlp_in*2, int(self.mlp_size))
                    fc2 = nn.Linear(int(self.mlp_size), num_classes)

                    h1 = nn.Linear(self.mlp_in*2, int(self.mlp_size))
                    h2 = nn.Linear(self.mlp_size, int(self.mlp_size) // 2)

                # weights = [41.0/387, 41.0/287, 41.0/64, 41.0/74, 41.0/41]
                # weights = torch.tensor(weights)
                # loss = nn.CrossEntropyLoss(weight=weights)

                module_dict['5'] = nn.Sequential(OrderedDict([
                                                ('dropout1',self.drop_out),
                                                ('fc1', fc1),
                                                ('dropout2', self.drop_out),
                                                ('relu', self.relu),
                                                ('fc2',fc2) ]))

                self.module_head = nn.Sequential(OrderedDict([
                                                ('h1', h1),
                                                ('relu', self.relu),
                                                ('h2', h2)]))

                # loss_dict['5'] = loss

            if "ESL" in dataset:
                num_classes = 2
                if self.max_num_class < num_classes:
                    self.max_num_class = num_classes

                if sub == True and mul == True:
                    fc1 = nn.Linear(self.mlp_in * 4, int(self.mlp_size * 2))
                    fc2 = nn.Linear(int(self.mlp_size * 2), num_classes)

                    h1 = nn.Linear(self.mlp_in * 4, int(self.mlp_size * 2))
                    h2 = nn.Linear(self.mlp_size * 2, int(self.mlp_size))

                if (sub == True and mul == False) or (sub == False and mul == True):
                    fc1 = nn.Linear(self.mlp_in * 3, int(self.mlp_size * 1.5))
                    fc2 = nn.Linear(int(self.mlp_size * 1.5), num_classes)

                    h1 = nn.Linear(self.mlp_in * 3, int(self.mlp_size * 1.5))
                    h2 = nn.Linear(self.mlp_size * 1.5, int(self.mlp_size))

                if sub == False and mul == False:
                    fc1 = nn.Linear(self.mlp_in * 2, int(self.mlp_size))
                    fc2 = nn.Linear(int(self.mlp_size), num_classes)

                    h1 = nn.Linear(self.mlp_in * 2, int(self.mlp_size))
                    h2 = nn.Linear(self.mlp_size, int(self.mlp_size) // 2)

                # weights = [41.0/387, 41.0/287, 41.0/64, 41.0/74, 41.0/41]
                # weights = torch.tensor(weights)
                # loss = nn.CrossEntropyLoss(weight=weights)

                module_dict['6'] = nn.Sequential(OrderedDict([
                    ('dropout1', self.drop_out),
                    ('fc1', fc1),
                    ('dropout2', self.drop_out),
                    ('relu', self.relu),
                    ('fc2', fc2)]))

                self.module_head = nn.Sequential(OrderedDict([
                    ('h1', h1),
                    ('relu', self.relu),
                    ('h2', h2)]))

                # loss_dict['5'] = loss

        self.module_dict = nn.ModuleDict(module_dict)
        self.loss_dict = nn.ModuleDict(loss_dict)

        self.task_weights = task_weights
        if self.task_weights != None:
            assert len(self.task_weights)==len(datasets), "Length of weight is difference number datasets: {}".format(len(self.task_weights))

    # def forward(self, sent, sent_mask, x_position, y_position, flag, sent_pos=None, views=None):
    #     batch_size = sent.size(0)
    #
    #     if self.finetune:
    #         output = self.roberta(sent, sent_mask)[2]
    #     else:
    #         with torch.no_grad():
    #             output = self.roberta(sent, sent_mask)[2]
    #
    #     output = torch.max(torch.stack(output[-4:], dim=0), dim=0)[0]
    #
    #     if sent_pos != None:
    #         # pos = self.pos_emb(sent_pos)
    #         output = torch.cat([output, self.pos_emb(sent_pos)], dim=2)
    #
    #     output = self.drop_out(output)
    #     output_A = torch.cat([output[i, x_position[i], :].unsqueeze(0) for i in range(0, batch_size)])
    #
    #     output_B = torch.cat([output[i, y_position[i], :].unsqueeze(0) for i in range(0, batch_size)])
    #
    #     if self.sub and self.mul:
    #         sub = torch.sub(output_A, output_B)
    #         mul = torch.mul(output_A, output_B)
    #         output = torch.cat([output_A, output_B, sub, mul], 1)
    #         del output_A, output_B, sub, mul
    #
    #     if self.sub==True and self.mul==False:
    #         sub = torch.sub(output_A, output_B)
    #         output = torch.cat([output_A, output_B, sub], 1)
    #         del output_A, output_B, sub
    #
    #     if self.sub==False and self.mul==True:
    #         mul = torch.mul(output_A, output_B)
    #         output = torch.cat([output_A, output_B, mul], 1)
    #         del output_A, output_B, mul
    #
    #     if self.sub==False and self.mul==False:
    #         output = torch.cat([output_A, output_B], 1)
    #         del output_A, output_B
    #
    #     if views is not None:
    #         # presentation = torch.reshape(presentation, (-1, views + 1, presentation.shape[-1]))
    #         output = self.module_head(output)
    #         output = F.normalize(output, dim=-1)
    #         return output
    #         # loss = self.loss_sc(feat, xy)
    #         # loss = self.loss_sc(feat)
    #         # return loss
    #     else:
    #         logits = []
    #         for i in range(0, batch_size):
    #             typ = str(flag[i].item())
    #             logit = self.module_dict[typ](output[i])
    #             logit = logit.unsqueeze(0)
    #             logits.append(logit)
    #         logits = torch.cat(logits,dim=0)
    #         return logits

    def forward(self, sent, sent_mask, x_position, y_position, xy,  flag, sent_pos=None, views=None):
        batch_size = sent.size(0)

        if self.finetune:
            # output = self.roberta(sent, sent_mask, output_hidden_states=True)[2]
            output = self.roberta(sent, sent_mask).last_hidden_state
        else:
            with torch.no_grad():
                output = self.roberta(sent, sent_mask)[2]

        # output = torch.max(torch.stack(output[-4:], dim=0), dim=0)[0]

        if sent_pos != None:
            pos = self.pos_emb(sent_pos)
            output = torch.cat([output, pos], dim=2)

        output = self.drop_out(output)
        output_A = torch.cat([output[i, x_position[i], :].unsqueeze(0) for i in range(0, batch_size)])

        output_B = torch.cat([output[i, y_position[i], :].unsqueeze(0) for i in range(0, batch_size)])

        if self.sub and self.mul:
            sub = torch.sub(output_A, output_B)
            mul = torch.mul(output_A, output_B)
            presentation = torch.cat([output_A, output_B, sub, mul], 1)
        if self.sub == True and self.mul == False:
            sub = torch.sub(output_A, output_B)
            presentation = torch.cat([output_A, output_B, sub], 1)
        if self.sub == False and self.mul == True:
            mul = torch.mul(output_A, output_B)
            presentation = torch.cat([output_A, output_B, mul], 1)
        if self.sub == False and self.mul == False:
            presentation = torch.cat([output_A, output_B], 1)

        if views is not None:
            # presentation = torch.reshape(presentation, (-1, views + 1, presentation.shape[-1]))
            head_feature = self.module_head(presentation)
            feat = F.normalize(head_feature, dim=-1)
            return feat
            # loss = self.loss_sc(feat, xy)
            # loss = self.loss_sc(feat)
            # return loss
        else:
            logits = []
            for i in range(0, batch_size):
                typ = str(flag[i].item())
                logit = self.module_dict[typ](presentation[i])
                logit = logit.unsqueeze(0)
                logits.append(logit)
            logits = torch.cat(logits, dim=0)
            return logits


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = "all"
        self.base_temperature = 0.07

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float()
        else:
            mask = mask.float()

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

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
        a = torch.arange(batch_size * anchor_count).view(-1, 1)

        if CUDA:
            mask = mask.cuda()
            a = a.cuda()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            a,
            0
        )

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss1 = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss1.view(anchor_count, batch_size).mean()
        if torch.isnan(loss):
            print(loss1.view(anchor_count, batch_size))

        return loss


# only symmetry negative sample
class ConLoss_S(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature):
        super(ConLoss_S, self).__init__()
        self.temperature = temperature
        self.contrast_mode = "all"
        self.base_temperature = 0.07

    def forward(self, features, view):
        num = features.shape[0] // 2
        features_pos = features[:num, :]
        features_neg = features[num:, :]

        anchor_dot_pos = torch.div(
            torch.matmul(features_pos, features_pos.T),
            self.temperature)
        # for numerical stability
        logits_max_pos, _ = torch.max(anchor_dot_pos, dim=1, keepdim=True)
        logits_pos = anchor_dot_pos - logits_max_pos.detach()

        anchor_dot_neg = torch.div(
            torch.matmul(features_pos, features_neg.T),
            self.temperature)
        # for numerical stability
        logits_max_neg, _ = torch.max(anchor_dot_neg, dim=1, keepdim=True)
        logits_neg = anchor_dot_neg - logits_max_neg.detach()

        batch_size = features_pos.shape[0] // view
        mask = torch.eye(batch_size, dtype=torch.float32)
        mask = torch.repeat_interleave(mask, view, dim=0)
        mask = torch.repeat_interleave(mask, view, dim=1)
        a = torch.arange(batch_size * view).view(-1, 1)
        if CUDA:
            mask = mask.cuda()
            a = a.cuda()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            a,
            0
        )
        mask_pos = mask * logits_mask
        pos_logits = torch.exp(logits_pos) * mask_pos
        pos = logits_pos * mask_pos
        neg_logits = torch.exp(logits_neg) * mask

        log_prob = pos - torch.log(pos_logits.sum(1, keepdim=True) + neg_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / mask.sum(1)

        loss1 = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss1.mean()

        return loss


# symmetry negative sample and random negative sample
class ConLoss_SR(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature):
        super(ConLoss_SR, self).__init__()
        self.temperature = temperature
        self.contrast_mode = "all"
        self.base_temperature = 0.07

    def forward(self, features, view):
        num = features.shape[0] // 2
        features_pos = features[:num, :]
        features_neg = features[num:, :]

        anchor_dot_pos = torch.div(
            torch.matmul(features_pos, features_pos.T),
            self.temperature)
        # for numerical stability
        logits_max_pos, _ = torch.max(anchor_dot_pos, dim=1, keepdim=True)
        logits_pos = anchor_dot_pos - logits_max_pos.detach()

        anchor_dot_neg = torch.div(
            torch.matmul(features_pos, features_neg.T),
            self.temperature)
        # for numerical stability
        logits_max_neg, _ = torch.max(anchor_dot_neg, dim=1, keepdim=True)
        logits_neg = anchor_dot_neg - logits_max_neg.detach()

        batch_size = features_pos.shape[0] // view
        mask = torch.eye(batch_size, dtype=torch.float32)
        mask = torch.repeat_interleave(mask, view, dim=0)
        mask = torch.repeat_interleave(mask, view, dim=1)
        mask_negInPos = torch.ones_like(mask) - mask
        a = torch.arange(batch_size * view).view(-1, 1)
        if CUDA:
            mask = mask.cuda()
            a = a.cuda()
            mask_negInPos = mask_negInPos.cuda()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            a,
            0
        )
        mask_pos = mask * logits_mask
        pos_logits = torch.exp(logits_pos) * mask_pos
        pos = logits_pos * mask_pos
        neg_logits = torch.exp(logits_neg) * mask
        mask_negInPos = mask_negInPos * logits_mask
        negInPos_logits = torch.exp(logits_pos) * mask_negInPos

        log_prob = pos - torch.log(pos_logits.sum(1, keepdim=True) + neg_logits.sum(1, keepdim=True) + negInPos_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / mask.sum(1)

        loss1 = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss1.mean()

        return loss


class ConLoss_R(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature):
        super(ConLoss_R, self).__init__()
        self.temperature = temperature
        self.contrast_mode = "all"
        self.base_temperature = 0.07

    def forward(self, features, view):
        num = features.shape[0] // 2
        features_pos = features[:num, :]
        features_neg = features[num:, :]

        anchor_dot_pos = torch.div(
            torch.matmul(features_pos, features_pos.T),
            self.temperature)
        # for numerical stability
        logits_max_pos, _ = torch.max(anchor_dot_pos, dim=1, keepdim=True)
        logits_pos = anchor_dot_pos - logits_max_pos.detach()

        anchor_dot_neg = torch.div(
            torch.matmul(features_pos, features_neg.T),
            self.temperature)
        # for numerical stability
        logits_max_neg, _ = torch.max(anchor_dot_neg, dim=1, keepdim=True)
        logits_neg = anchor_dot_neg - logits_max_neg.detach()

        batch_size = features_pos.shape[0] // view
        mask = torch.eye(batch_size, dtype=torch.float32)
        mask = torch.repeat_interleave(mask, view, dim=0)
        mask = torch.repeat_interleave(mask, view, dim=1)
        mask_negInPos = torch.ones_like(mask) - mask
        a = torch.arange(batch_size * view).view(-1, 1)
        if CUDA:
            mask = mask.cuda()
            a = a.cuda()
            mask_negInPos = mask_negInPos.cuda()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            a,
            0
        )
        mask_pos = mask * logits_mask
        pos_logits = torch.exp(logits_pos) * mask_pos
        pos = logits_pos * mask_pos
        neg_logits = torch.exp(logits_neg) * mask
        mask_negInPos = mask_negInPos * logits_mask
        negInPos_logits = torch.exp(logits_pos) * mask_negInPos

        log_prob = pos - torch.log(pos_logits.sum(1, keepdim=True) + negInPos_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / mask.sum(1)

        loss1 = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss1.mean()

        return loss

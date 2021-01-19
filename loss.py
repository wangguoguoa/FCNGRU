#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

##############  double branch #################
class OhemNegLoss(nn.Module):
    def __init__(self, a1, a2, device, thresh=0.7):
        super(OhemNegLoss, self).__init__()
        a1 = torch.tensor(a1)
        self.a1 = a1.to(device)
        a2 = torch.tensor(a2)
        self.a2 = a2.to(device)
        thresh = torch.tensor(thresh)
        self.thresh = thresh.to(device)
        self.criteria1 = nn.MSELoss()
        self.criteria2 = nn.BCELoss(reduction='none')
        self.criteria3 = nn.BCELoss()

    def forward(self, label_p, label_t, denselabel_p, denselabel_t):
        label_p = label_p.view(-1)
        label_t = label_t.view(-1)
        denselabel_p = denselabel_p.view(-1)
        denselabel_t = denselabel_t.view(-1)
        # hard negative example mining
        index_pos = (denselabel_t == 1)
        index_neg = (denselabel_t == 0)
        denselabel_p_pos = denselabel_p[index_pos]
        denselabel_t_pos = denselabel_t[index_pos]
        denselabel_p_neg = denselabel_p[index_neg]
        denselabel_t_neg = denselabel_t[index_neg]

        loss_pos = self.criteria2(denselabel_p_pos, denselabel_t_pos)
        loss_neg = self.criteria2(denselabel_p_neg, denselabel_t_neg)
        loss_neg, _ = torch.sort(loss_neg, descending=True)
        number_neg = int(self.thresh*loss_neg.numel())
        loss_neg = loss_neg[:number_neg]
        loss = torch.mean(loss_pos) + torch.mean(loss_neg)
        # classifier loss + dense prediction loss
        loss_total = self.a1*self.criteria1(label_p, label_t) + self.a2*loss
        return loss_total

# ########## single branch ##############
# class OhemNegLoss(nn.Module):
#     def __init__(self, device, thresh=0.3):
#         super(OhemNegLoss, self).__init__()
#         thresh = torch.tensor(thresh)
#         self.thresh = thresh.to(device)
#         self.criteria = nn.BCELoss(reduction='none')
#
#     def forward(self, denselabel_p, denselabel_t):
#         # hard negative example mining
#         denselabel_p_v = denselabel_p.view(-1)
#         denselabel_t_v = denselabel_t.view(-1)
#         index_pos = (denselabel_t_v == 1)
#         index_neg = (denselabel_t_v == 0)
#         denselabel_p_pos = denselabel_p_v[index_pos]
#         denselabel_t_pos = denselabel_t_v[index_pos]
#         denselabel_p_neg = denselabel_p_v[index_neg]
#         denselabel_t_neg = denselabel_t_v[index_neg]
#
#         loss_pos = self.criteria(denselabel_p_pos, denselabel_t_pos)
#         loss_neg = self.criteria(denselabel_p_neg, denselabel_t_neg)
#         loss_neg, _ = torch.sort(loss_neg, descending=True)
#         number_neg = int(self.thresh*loss_neg.numel())
#         loss_neg = loss_neg[:number_neg]
#         loss = torch.mean(loss_pos) + torch.mean(loss_neg)
#         return loss

#
class OhemLoss(nn.Module):
    def __init__(self, a1, a2, device):
        super(OhemLoss, self).__init__()
        a1 = torch.tensor(a1)
        self.a1 = a1.to(device)
        a2 = torch.tensor(a2)
        self.a2 = a2.to(device)
        self.criteria1 = nn.MSELoss()#huigui
        self.criteria2 = nn.BCELoss()

    def forward(self, label_p, label_t, denselabel_p, denselabel_t):
        label_p = label_p.view(-1)
        label_t = label_t.view(-1)
        loss_total = self.a1 * self.criteria1(label_p, label_t) + self.a2 * self.criteria2(denselabel_p, denselabel_t)
        return loss_total


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, labels):
        factor_p = torch.pow(1.-logits, self.gamma)
        factor_n = torch.pow(logits, self.gamma)
        if self.alpha is None:
            loss = -labels * factor_p * torch.log(logits) \
                   - (1 - labels) * factor_n * torch.log(1 - logits)
        else:
            loss = - self.alpha * labels * factor_p * torch.log(logits) \
                   - (1 - self.alpha) * (1 - labels) * factor_n * torch.log(1 - logits)
        loss = torch.mean(loss)
        return loss


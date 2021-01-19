import os
import math
import datetime
import numpy as np
import os.path as osp
from utils import plotandsave, label_accuracy_score
from sklearn.metrics import r2_score
from scipy import stats

import torch

#############  double-branch ###########################
class Trainer(object):
    """build a trainer"""
    def __init__(self, model, optimizer, criterion, device, checkpoint, start_epoch, max_epoch,
                 train_loader, test_loader, lr_policy, plot=False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.plot = plot
        self.checkpoint = checkpoints
        if not osp.exists(self.checkpoint):
            os.mkdir(self.checkpoint)
        self.LR_policy = lr_policy
        self.log_headers = [
            'epoch',
            'valid/pear',
            'valid/iou'
        ]
        with open(osp.join(self.checkpoint, 'log.csv'), 'w') as f:
            f.write('\t'.join(self.log_headers) + '\n')
        self.epoch = 0
        self.pear_best = -1
        # self.spr_best = 0
        self.iou_best = 0
        self.state_best = None

    def train(self):
        """training the model"""
        self.model.to(self.device)
        self.criterion.to(self.device)
        for epoch in range(self.start_epoch, self.max_epoch):
            # set training mode during the training process
            self.model.train()
            self.epoch = epoch
            # self.LR_policy.step() # for cosine learning strategy
            for i_batch, sample_batch in enumerate(self.train_loader):
                X_data = sample_batch["data"].float().to(self.device)
                label = sample_batch["label"].float().to(self.device)
                denselabel = sample_batch["denselabel"].float().to(self.device)
                self.optimizer.zero_grad()
                label_p, denselabel_p = self.model(X_data)
                loss = self.criterion(label_p, label, denselabel_p, denselabel)
                if np.isnan(loss.item()):
                    raise ValueError('loss is nan while training')
                loss.backward()
                self.optimizer.step()
                print("epoch/i_batch: {}/{}---loss: {:.4f}---lr: {:.4f}".format(self.epoch, i_batch, loss.item(), self.optimizer.param_groups[0]['lr']))
            # validation and save the model with higher accuracy
            pear, iou = self.test()
            # record some key results
            with open(osp.join(self.checkpoint, 'log.csv'), 'a') as f:
                log = [self.epoch] + [pear] + [iou]
                log = map(str, log)
                f.write('\t'.join(log) + '\n')

        return self.pear_best,  self.iou_best, self.state_best

    def test(self):
        """validate the performance of the trained model."""
        self.model.eval()
        label_p_all = []
        label_t_all = []
        seg_p_all = []
        seg_t_all = []
        for i_batch, sample_batch in enumerate(self.test_loader):
            X_data = sample_batch["data"].float().to(self.device)
            label = sample_batch["label"].float().to(self.device)
            denselabel = sample_batch["denselabel"].float().to(self.device)
            with torch.no_grad():
                label_p, denselabel_p = self.model(X_data)

            label_p_all.append(label_p.view(-1).data.cpu().numpy()[0])
            label_t_all.append(label.view(-1).data.cpu().numpy()[0])
            seg_p_all.append(denselabel_p.view(-1).data.cpu().numpy() > 0.7)
            seg_t_all.append(denselabel.view(-1).data.cpu().numpy())

        iou = label_accuracy_score(seg_t_all, seg_p_all, n_class=2)
        pear, _ = np.array(stats.pearsonr(label_t_all, label_p_all),dtype='float32')
        if (self.pear_best + self.iou_best) < (pear + iou):
            self.pear_best = pear
            self.iou_best = iou
            self.state_best = self.model.state_dict()
        print("pear: {:.3f}\tiou: {:.3f}\n".format(pear, iou))
        return pear, iou


#####################   single-branch #########################
# class Trainer(object):
#     """build a trainer"""
#
#     def __init__(self, model, optimizer, criterion, device, checkpoint, start_epoch, max_epoch,
#                  train_loader, test_loader, lr_policy, plot=False):
#         self.model = model
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.device = device
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.start_epoch = start_epoch
#         self.max_epoch = max_epoch
#         self.plot = plot
#         self.checkpoint = checkpoint
#         if not osp.exists(self.checkpoint):
#             os.mkdir(self.checkpoint)
#         self.LR_policy = lr_policy
#         self.log_headers = [
#             'epoch',
#             'valid/iou'
#         ]
#         with open(osp.join(self.checkpoint, 'log.csv'), 'w') as f:
#             f.write('\t'.join(self.log_headers) + '\n')
#         self.epoch = 0
#         # self.pear_best = -1
#         self.iou_best = 0
#         self.state_best = None
#
#     def train(self):
#         """training the model"""
#         self.model.to(self.device)
#         self.criterion.to(self.device)
#         for epoch in range(self.start_epoch, self.max_epoch):
#             # set training mode during the training process
#             self.model.train()
#             self.epoch = epoch
#             # self.LR_policy.step() # for cosine learning strategy
#             for i_batch, sample_batch in enumerate(self.train_loader):
#                 X_data = sample_batch["data"].float().to(self.device)
#                 # label = sample_batch["label"].float().to(self.device)
#                 denselabel = sample_batch["denselabel"].float().to(self.device)
#                 self.optimizer.zero_grad()
#                 # label_p, denselabel_p = self.model(X_data)
#                 denselabel_p = self.model(X_data)
#                 # loss = self.criterion(label_p, label, denselabel_p, denselabel)
#                 loss = self.criterion(denselabel_p, denselabel)
#                 if np.isnan(loss.item()):
#                     raise ValueError('loss is nan while training')
#                 loss.backward()
#                 self.optimizer.step()
#                 print("epoch/i_batch: {}/{}---loss: {:.4f}---lr: {:.4f}".format(self.epoch, i_batch, loss.item(),
#                                                                                 self.optimizer.param_groups[0]['lr']))
#             # validation and save the model with higher accuracy
#             # pear, iou = self.test()
#             iou = self.test()
#             # record some key results
#             with open(osp.join(self.checkpoint, 'log.csv'), 'a') as f:
#                 # log = [self.epoch] + [pear] + [iou]
#                 log = [self.epoch] + [iou]
#                 log = map(str, log)
#                 f.write('\t'.join(log) + '\n')
#
#         # return self.pear_best, self.iou_best, self.state_best
#         return self.iou_best, self.state_best
#
#     def test(self):
#         """validate the performance of the trained model."""
#         self.model.eval()
#         label_p_all = []
#         label_t_all = []
#         seg_p_all = []
#         seg_t_all = []
#         for i_batch, sample_batch in enumerate(self.test_loader):
#             X_data = sample_batch["data"].float().to(self.device)
#             # label = sample_batch["label"].float().to(self.device)
#             denselabel = sample_batch["denselabel"].float().to(self.device)
#             with torch.no_grad():
#                 # label_p, denselabel_p = self.model(X_data)
#                 denselabel_p = self.model(X_data)
#
#             # label_p_all.append(label_p.view(-1).data.cpu().numpy()[0])
#             # label_t_all.append(label.view(-1).data.cpu().numpy()[0])
#             seg_p_all.append(denselabel_p.view(-1).data.cpu().numpy() > 0.7)
#             seg_t_all.append(denselabel.view(-1).data.cpu().numpy())
#
#         iou = label_accuracy_score(seg_t_all, seg_p_all, n_class=2)
#         # pear, _ = np.array(stats.pearsonr(label_t_all, label_p_all),dtype='float32')
#         # if (self.pear_best + self.iou_best) < (pear + iou):
#         #     self.pear_best = pear
#         #     self.iou_best = iou
#         #     self.state_best = self.model.state_dict()
#         # print("pear: {:.3f}\tiou: {:.3f}\n".format(pear, iou))
#         # return pear, iou
#
#         # pear, _ = np.array(stats.pearsonr(label_t_all, label_p_all),dtype='float32')
#         if self.iou_best < iou:
#             self.iou_best = iou
#             self.state_best = self.model.state_dict()
#         print("iou: {:.3f}\n".format(iou))
#         return iou

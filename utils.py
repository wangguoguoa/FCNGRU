#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import math
import warnings


import numpy as np
from abc import ABCMeta, abstractmethod
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
#  Learning rate policy
# -----------------------------------------------------------------------------


class BaseLR():
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_lr(self, cur_iter): pass


class PolyLR(BaseLR):
    """Learning Rate Scheduler

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ power``

    """
    def __init__(self, start_lr, lr_power, total_iters):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = 1.0 * total_iters

    def __call__(self, optimizer, cur_iter):
 
        lr = self.start_lr * math.pow((1 - 1.0 * cur_iter / self.total_iters),
                                      self.lr_power)

        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10  


# -----------------------------------------------------------------------------
#  plot and save
# -----------------------------------------------------------------------------


def plotandsave(auc_records, prauc_records, loss_records, file_path):
    """
    plot the training process and save it.
    :param auc_records:
    :param prauc_records:
    :param loss_records:
    :return:
    """
    train_len = len(auc_records)
    x = np.asarray(list(range(train_len))) + 1

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, auc_records, 'r.-', x, prauc_records, 'b.-')
    ax1.legend(['auc', 'prauc'], loc='upper left')
    ax1.set_title('training/validation process')
    ax1.set_ylabel('auc/prauc')
    ax1.set_xlabel('epoch')
    ax1.set_ylim([0, 1])
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, loss_records, 'g.--')
    ax2.legend('loss', loc='lower left')
    ax2.set_ylabel('loss')
    #plt.show()
    plt.savefig(file_path, format='png')

# -----------------------------------------------------------------------------
#  compute the number of paramters
# -----------------------------------------------------------------------------


def get_n_params(parameters):
    pp = 0
    for p in parameters:
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

# -----------------------------------------------------------------------------
# IOU Evaluation
# -----------------------------------------------------------------------------


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    for ind_class in range(n_class):
        print('===>' + 'label {}'.format(ind_class) + ':\t' + str(round(iu[ind_class] * 100, 2)))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return mean_iu

# -----------------------------------------------------------------------------
# Dict
# -----------------------------------------------------------------------------


def Dict(infile):
    dict={}
    lines = open(infile).readlines()
    for line in lines:
        line_split = line.strip().split()
        dict[line_split[0]] = int(line_split[1])
    return dict


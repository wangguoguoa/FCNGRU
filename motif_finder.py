#!/usr/bin/python

import os
import sys
import time
import argparse
import math
import numpy as np
import os.path as osp

import torch
from torch.utils.data import DataLoader

# custom functions defined by user
from datasets import EPIDataSetTrain, EPIDataSetTest
from utils import Dict
import torch.nn as nn
import torch.nn.functional as F


def upsample(x, out_size):
    return F.interpolate(x, size=out_size, mode='linear', align_corners=False)


def bn_relu_conv(in_, out_, kernel_size=3, stride=1, bias=False):
    padding = kernel_size // 2
    return nn.Sequential(nn.BatchNorm1d(in_),
                         nn.ReLU(inplace=True),
                         nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))





#############  single branch ##############
# class FCNGRU(nn.Module):
#     """FCN for motif mining"""
#     def __init__(self, motiflen=13):
#         super(FCNGRU, self).__init__()
#         # encode process
#         self.conv1 = nn.Conv1d(in_channels=4, out_channels=64,  kernel_size=motiflen, padding=motiflen//2)
#         self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, padding=3)
#         self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
#         self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
#         # decode process
#         self.gru = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
#         self.gru_drop = nn.Dropout(p=0.5)
#         # self.aap = nn.AdaptiveAvgPool1d(1)
#         self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
#         self.blend2 = bn_relu_conv(64, 4, kernel_size=3)
#         self.blend1 = bn_relu_conv(4, 1, kernel_size=3)
#         # regression head
#         # general functions
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(p=0.2)
#         self._init_weights()
#
#     def _init_weights(self):
#         """Initialize the new built layers"""
#         for layer in self.modules():
#             if isinstance(layer, (nn.Conv1d, nn.Linear)):
#                 # nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
#                 nn.init.xavier_uniform_(layer.weight)
#                 if layer.bias is not None:
#                     nn.init.constant_(layer.bias, 0)
#             elif isinstance(layer, nn.BatchNorm1d):
#                 nn.init.constant_(layer.weight, 1)
#                 nn.init.constant_(layer.bias, 0)
#
#     def forward(self, data):
#         """Construct a new computation graph at each froward"""
#         b, _, _ = data.size()
#         # encode process
#         skip1 = data
#         out1 = self.conv1(data)
#         out1 = self.relu(out1)
#         score = out1
#         out1 = self.pool1(out1)
#         out1 = self.dropout(out1)
#         skip2 = out1
#         out1 = self.conv2(out1)
#         out1 = self.relu(out1)
#         out1 = self.pool2(out1)
#         out1 = self.dropout(out1)
#         skip3 = out1
#         out1 = self.conv3(out1)
#         out1 = self.relu(out1)
#         out1 = self.pool3(out1)
#         out1 = self.dropout(out1)
#
#         skip4 = out1.permute(0, 2, 1)
#         up4, _ = self.gru(skip4)
#         up4 = self.gru_drop(up4)
#         up4 = up4.permute(0, 2, 1)
#
#         # decode process
#
#         up3 = upsample(up4, skip3.size()[-1])
#         up3 = up3 + skip3
#         up3 = self.blend3(up3)
#         up2 = upsample(up3, skip2.size()[-1])
#         up2 = up2 + skip2
#         up2 = self.blend2(up2)
#         up1 = upsample(up2, skip1.size()[-1])
#         up1 = up1 + skip1
#         up1 = self.blend1(up1)
#         out_dense = self.sigmoid(up1)
#         out_dense = out_dense.view(b, -1)
#
#         # regression
#
#
#         return  out_dense,score[0]



################# double branch ############
class FCNGRU(nn.Module):
    """FCN for motif mining"""
    def __init__(self, motiflen=13):
        super(FCNGRU, self).__init__()
        # encode process
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=motiflen, padding=motiflen//2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, padding=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # decode process
        self.gru = nn.GRU(input_size=64, hidden_size=64, batch_first=True)
        self.gru_drop = nn.Dropout(p=0.5)
        # self.aap = nn.AdaptiveAvgPool1d(1)
        self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend2 = bn_relu_conv(64, 4, kernel_size=3)
        self.blend1 = bn_relu_conv(4, 1, kernel_size=3)
        # regression head
        c_in = 256
        self.linear1 = nn.Linear(c_in, 64)
        self.drop = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(64, 1)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                # nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        # encode process
        skip1 = data
        out1 = self.conv1(data)
        out1 = self.relu(out1)
        score = out1
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)

        skip4 = out1.permute(0, 2, 1)
        up4, _ = self.gru(skip4)
        up4 = self.gru_drop(up4)
        up4 = up4.permute(0, 2, 1)

        # decode process

        up3 = upsample(up4, skip3.size()[-1])
        up3 = up3 + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        up2 = up2 + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        up1 = up1 + skip1
        up1 = self.blend1(up1)
        out_dense = self.sigmoid(up1)
        out_dense = out_dense.view(b, -1)

        # regression
        out2 = skip4.reshape(b, -1)
        out2 = self.linear1(out2)
        out2 = self.relu(out2)
        out2 = self.drop(out2)
        out_regression = self.linear2(out2)

        return out_dense, score[0]


def extract(denselabel):
    position = 0
    maxcount = 0
    num = 1
    for i in range(len(denselabel) - WINDOW + 1):
        subseq = denselabel[i:(i+WINDOW)]
        count = np.sum(subseq)
        if maxcount < count:
            maxcount = count
            position = i
            num = 1
        elif maxcount == count:
            position += i
            num += 1
    start = int(np.ceil(position / num))
    end = start + WINDOW
    return start, end


def find_count(index):
    dict = {}
    for e in index:
        if str(e) in dict.keys():
            dict[str(e)] += 1
        else:
            dict[str(e)] = 1
    count = 0
    for k, v in dict.items():
        if count < v:
            count = v
            ip = k
    return int(ip)

def motif_all(device, model, state_dict, train_loader, test_loader, outdir, thres):
    # loading model parameters
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    motif_data = [0.] * kernel_num
    for i_batch, sample_batch in enumerate(train_loader):
        X_data = sample_batch["data"].float().to(device)
        with torch.no_grad():
            denselabel_p, score_p = model(X_data)
        denselabel_p = np.concatenate((np.zeros((motifLen//2,)), denselabel_p.view(-1).data.cpu().numpy(), np.zeros(motifLen//2,)))
        denselabel_p = (denselabel_p > thres)
        start, end = extract(denselabel_p)
        data = X_data[0].data.cpu().numpy()
        data = np.concatenate((np.zeros((4, motifLen//2)), data, np.zeros((4, motifLen//2))), axis=1)
        score_p = score_p.data.cpu().numpy()
        score_p = score_p[:, start:end]
        max_index = np.argmax(score_p, axis=1)
        for i in range(kernel_num):
            index = max_index[i]
            index += start
            # if index + motifLen > 35:
            #     print(index+motifLen)
            #     continue
            data_slice = data[:, index:(index+motifLen)]
            motif_data[i] += data_slice

    pfm = compute_pfm(motif_data)
    writeFile(pfm, 'train_all', outdir)
    # for test data
    motif_data = [0.] * kernel_num
    for i_batch, sample_batch in enumerate(test_loader):
        X_data = sample_batch["data"].float().to(device)
        with torch.no_grad():
            denselabel_p, score_p = model(X_data)
        denselabel_p = np.concatenate((np.zeros((motifLen//2,)), denselabel_p.view(-1).data.cpu().numpy(), np.zeros(motifLen//2,)))
        denselabel_p = (denselabel_p > thres)
        start, end = extract(denselabel_p)
        data = X_data[0].data.cpu().numpy()
        data = np.concatenate((np.zeros((4, motifLen//2)), data, np.zeros((4, motifLen//2))), axis=1)
        score_p = score_p.data.cpu().numpy()
        score_p = score_p[:, start:end]
        max_index = np.argmax(score_p, axis=1)
        for i in range(kernel_num):
            index = max_index[i]
            index += start
            data_slice = data[:, index:(index + motifLen)]
            motif_data[i] += data_slice

    pfm = compute_pfm(motif_data)
    writeFile(pfm, 'test_all', outdir)


def compute_pfm(motifs):
    pfms = []
    for motif in motifs:
        sum_ = np.sum(motif, axis=0)
        pfm = motif / sum_
        pfms.append(pfm)

    return pfms


def writeFile(pfm, flag, outdir):
    out_f = open(outdir + '/{}_pfm.txt'.format(flag), 'w')
    out_f.write("MEME version 5.1.1\n\n")
    out_f.write("ALPHABET= ACGT\n\n")
    out_f.write("strands: + -\n\n")
    out_f.write("Background letter frequencies\n")
    out_f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")
    for i in range(len(pfm)):
        out_f.write("MOTIF " + "{}\n".format(i+1))
        out_f.write("letter-probability matrix: alength= 4 w= {} nsites= {}\n".format(motifLen, motifLen))
        current_pfm = pfm[i]
        for col in range(current_pfm.shape[1]):
            for row in range(current_pfm.shape[0]):
                out_f.write("{:.4f} ".format(current_pfm[row, col]))
            out_f.write("\n")
        out_f.write("\n")
    out_f.close()


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="FCN for motif location")

    parser.add_argument("-d", dest="data_dir", type=str, default=None,
                        help="A directory containing the training data.")
    parser.add_argument("-n", dest="name", type=str, default=None,
                        help="The name of a specified data.")

    parser.add_argument("-t", dest="thres", type=float, default=0.5,
                        help="threshold value.")
    parser.add_argument("-g", dest="gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to save snapshots of the model.")
    parser.add_argument("-o", dest="outdir", type=str, default='./motifs/',
                        help="Where to save experimental results.")

    return parser.parse_args()


args = get_args()
motifLen_dict = Dict(os.getcwd() + '/motifLen.txt')
motifLen = motifLen_dict[args.name]
kernel_num = 64
WINDOW = 25

def main():
    """Create the model and start the training."""
    if torch.cuda.is_available():
        if len(args.gpu.split(',')) == 1:
            device = torch.device("cuda:" + args.gpu)
        else:
            device = torch.device("cuda:" + args.gpu.split(',')[0])
    else:
        device = torch.device("cpu")
    Data = np.load(osp.join(args.data_dir, '%s_data.npz' % args.name))
    seqs, label, denselabel = Data['data'], Data['label'], Data['denselabel']
    ##
    pos_index = (label == 1)
    seqs = seqs[pos_index]
    denselabel = denselabel[pos_index]
    ##
    cv_num = 5
    interval = int(len(seqs) / cv_num)
    index = range(len(seqs))
    # 5-fold cross validation
    for cv in range(1):
        index_test = index[cv * interval:(cv + 1) * interval]
        index_train = list(set(index) - set(index_test))
        # build training data generator
        data_tr = seqs[index_train]
        label_tr = label[index_train]
        denselabel_tr = denselabel[index_train]
        train_data = EPIDataSetTrain(data_tr, label_tr, denselabel_tr)
        train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=1)
        # build test data generator
        data_te = seqs[index_test]
        label_te = label[index_test]
        denselabel_te = denselabel[index_test]
        test_data = EPIDataSetTest(data_te, label_te,denselabel_te)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
        # Load weights
        checkpoint_file = osp.join(args.checkpoint, 'model_best%d.pth' % cv)
        chk = torch.load(checkpoint_file)
        state_dict = chk['model_state_dict']
        model = FCNGRU(motiflen=motifLen)
        motif_all(device, model, state_dict, train_loader, test_loader, args.outdir, args.thres)



if __name__ == "__main__":
    main()


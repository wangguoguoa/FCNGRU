import os.path as osp
import os,sys
import argparse
import itertools
import numpy as np

outdir = '/home/daguoguo/wsgg/FC/FCNMotifR_normalization/invitro/'
newdatadata = '/home/daguoguo/wsgg/FC/FCNMotifR_normalization/newdata/'


host_dict = {}
dirs = os.listdir(outdir)
for file in dirs:
    dir1 = osp.join(outdir, '%s' % file)
    dir2 = os.listdir(dir1)
    f_allhost = open(osp.join(dir1,'%s_v1_deBruijn.txt' % file), 'r')
    lines = f_allhost.readlines()
    for line in lines:
        line_list = line.strip().split()
        name = line_list[1]
        host = line_list[0]
        if host is not '' and float(host) > 0:
            if name in host_dict:
                host_dict.get(name).append(host)
            else:
                host_dict.setdefault(name,[]).append(host)

for file in dirs:
    print('Experiment on %s dataset' % file)
    seqs = []
    labels = []
    dir1 = osp.join(outdir, '%s' % file)
    dir2 = os.listdir(dir1)
    f = open(osp.join(dir1,'%s_v1_deBruijn.txt' % file), 'r+')
    for line in f:
        line_split = line.strip().split()
        if line_split[1] in host_dict:
            data = host_dict.get(line_split[1])
            median = np.median(np.asarray(data, dtype=np.float32))
            line_split[0] = float(line_split[0])
            line_split[0] /= median
            if line_split[0] > 0:
                seqs.append(line_split[1][:35])
                labels.append(float(line_split[0]))
            else:
                print('%s dataset has negative value' % file)
        else:
            print("cant find  same seq in directory")
    labels = np.around(labels, decimals=2)
    a = np.array([labels,seqs])
    a = a.transpose(1, 0)
    # path = osp.join(outdir, '%s/%s_new_v1_deBruijn' % file)
    np.savetxt(dir1 + '/%s_new_v1_deBruijn.txt' % file, a, fmt='%s %s', delimiter=' ')
    f.close()
    # labels = np.asarray(labels, dtype=np.float32)
    # median = np.median(labels)
    # index = labels > np.quantile(labels, 0.5)
    # labels /= median
    # labels = np.around(labels, decimals=2)


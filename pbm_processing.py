# coding:utf-8
import os.path as osp
import os
import argparse
import itertools
import numpy as np

mapper = {'A': [1., 0., 0., 0.],
          'C': [0., 1., 0., 0.],
          'G': [0., 0., 1., 0.],
          'T': [0., 0., 0., 1.],
          'N': [0., 0., 0., 0.]}


# embed sequences into one-hot vector
def embed(sequences):
    embed_vector = []
    for seq in sequences:
        mat = []
        for element in seq:
            if element in mapper.keys():
                mat.append(mapper[element])
            else:
                print('invalid character!')
                sys.exit(1)
        embed_vector.append(mat)
    embed_vector = np.asarray(embed_vector, dtype=np.float32)
    embed_vector = embed_vector.transpose((0, 2, 1))
    # embed_vector = embed_vector.transpose((0, 1, 2))
    return embed_vector


def denselabel(data, pwmfile):
    """data: N*4*L, pwm: 4*k"""
    pwm = []
    with open(pwmfile, 'r') as f:
        for line in f:
            line_split = line.strip().split()
            pwm.append([float(i) for i in line_split])
    pwm = np.asarray(pwm)
    N, _, L = data.shape
    _, k = pwm.shape
    labels = []
    best_values = []
    for i in range(N):
        data_row = data[i]
        records = np.zeros(L-k+1)
        for j in range(L-k+1):
            records[j] = np.sum(data_row[:, j:(j+k)] * pwm)
        best_index = np.argmax(records)
        best_values.append(np.max(records))
        label_row = np.zeros(L)
        label_row[best_index:(best_index+k)] = 1.
        labels.append(label_row)

    return np.asarray(labels, dtype=np.float32), np.asarray(best_values)  # N*L

def get_fold(root, target):
    items = os.listdir(root)
    for item in items:
        path = os.path.join(root, item)
        if os.path.isdir(path):
            print('[-]', path)
            get_fold(path, target)
        elif path.split('/')[-1] == target:
            print('[+]', path)
        else:
            print('[!]', path)

def get_data(infile, pwmfile):
    seqs = []
    labels = []
    f = open(infile, 'r')
    for line in f:
        line_split = line.strip().split()
        seqs.append(line_split[1])
        labels.append(float(line_split[0]))
    f.close()
    # labels = np.asarray(labels, dtype=np.float32)
    # median = np.median(labels)
    # index = labels > np.quantile(labels, 0.25)
    # labels /= median
    labels = np.around(labels, decimals=2)
    data = embed(seqs)
    # labels_by_index = labels[index]
    # data_by_index = data[index]
    labels_by_index = labels
    data_by_index = data
    labels_by_pwm, best_values = denselabel(data_by_index, pwmfile)

    return data_by_index, labels_by_index, labels_by_pwm, best_values


def get_args():
    parser = argparse.ArgumentParser(description="pre-process PBM data.")
    parser.add_argument("-if", dest="infile", type=str, default='')
    parser.add_argument("-pwm", dest="pwm", type=str, default='')
    parser.add_argument("-n", dest="name", type=str, default='')
    parser.add_argument("-o", dest="outdir", type=str, default='')


    return parser.parse_args()


def main():
    params = get_args()
    infile = params.infile
    pwmfile = params.pwm
    name = params.name
    outdir = params.outdir
    out_dir = osp.join(params.outdir, '%s/data/' % name)
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    print('Experiment on %s dataset' % name)

    print('Loading seq data...')
    data_by_index, labels_by_index, labels_by_pwm, best_values = get_data(infile, pwmfile)

    np.savez(out_dir+'%s_data.npz' % name, data=data_by_index, label=labels_by_index, denselabel=labels_by_pwm)


if __name__ == '__main__': main()


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import time
import json
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import pandas as pd
from datetime import timedelta
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np
import sys

PEP = 0.93


def expToDict(fp):
    exp_dic = dict()
    scan = []
    for line_id, line in enumerate(fp):
        line = line.strip()
        if line_id % 5 == 0:
            if len(scan) != 0:
                exp_dic[key] = scan
            scan = []
            key = line
        else:
            junk = line.split(' ')
            x = []
            if len(junk) > 1:
                for each in junk:
                    x.append(float(each))  # timediff=3.96
                scan.append(x)
            else:
                scan.append([])
    exp_dic[key] = scan  # timediff=8.97
    return exp_dic


def theoryToDict(fp):
    theory_dic = dict()
    scan = []
    for line_id, line in enumerate(fp):
        line = line.strip()
        if line_id % 7 == 0:
            if len(scan) != 0:
                theory_dic[key] = scan
            scan = []
            key = line
        else:
            junk = line.split(' ')
            x = []
            if len(junk) > 1:
                for each in junk:
                    x.append(float(each))
                scan.append(x)
            else:
                scan.append([])
    theory_dic[key] = scan  # timediff=29.8
    return theory_dic


def featureToDict(fp):
    feature = fp.read().strip().split('\n\n')
    feature_dic = dict()
    for scan in feature:
        lines = scan.strip().split('\n')
        pepidx = lines[0]
        feature = lines[2].strip().split()
        # feature_dic[pepidx] = feature  #timediff=0.23
        # feature_dic[pepidx] = np.asarray(feature,dtype=float) #timediff=1.14
        feature_dic[pepidx] = [float(x) for x in feature]  # timediff=0.55

    fp.close()
    return feature_dic


def LabelToDict(fp):
    sample = fp.read().strip().split('\n')[1:]
    label_dic = dict()
    for scan in sample:
        scan = scan.strip().split(',')
        idx = '{0}_{1}_{2}_{3}'.format(str(scan[2]), str(scan[3]), str(scan[4]), str(scan[5]))
        weight = float(scan[1])
        pep = float(scan[8])
        if scan[0] == 'True':
            label = 1
        else:
            label = 0
        label_dic[idx] = [pep, label, weight]

    return label_dic


def readData(expPrefix, theoryPrefix, featurePrefix, LabelPrefix, filenum):
    L = []
    Y = []
    weight = []
    for i in range(1, int(filenum) + 1):
        filename = theoryPrefix + '_' + str(i) + '.txt'
        f = open(filename)
        D_theory = theoryToDict(f)
        filename = featurePrefix + '_' + str(i) + '.txt'
        f = open(filename)
        D_feature = featureToDict(f)
        filename = expPrefix + '_' + str(i) + '.txt'
        f = open(filename)
        D_exp = expToDict(f)
        filename = LabelPrefix + '_' + str(i) + '.csv'
        f = open(filename)
        D_Label = LabelToDict(f)

        for j in D_Label.keys():
            if D_Label[j][0] < PEP:
                l = []
                if j[:-4] not in D_exp.keys():
                    continue
                else:
                    l.append(D_exp[j[:-4]])
                    l.append(D_theory[j])
                    l.append(D_feature[j])
                    L.append(l)
                    Y.append(D_Label[j][1])
                    weight.append(D_Label[j][2])
        D_theory = dict()
        D_exp = dict()
        D_feature = dict()
        D_Label = dict()

    return L, Y, weight


class DefineDataset(Data.Dataset):
    def __init__(self, X_index, L, Y, weight):
        self.X_index = X_index
        self.L = L
        self.Y = Y
        self.weight = weight

    def __len__(self):
        return len(self.X_index)

    def __getitem__(self, idx):
        idx = self.X_index[idx]
        expvec = self.L[idx][0]
        theoryvec = self.L[idx][1]
        addFeatvec = self.L[idx][2]

        width = 0.5
        count = 0
        construction = []
        for i in range(10):
            s = []
            for j in range(3600):
                s.append(0)
            construction.append(s)

        for construction_id, chargeMZ in enumerate(expvec):
            # m/z by charge is not exist
            if len(chargeMZ) == 0:
                continue
            for line_id, line in enumerate(chargeMZ):
                if line_id % 2 == 0:
                    matrix_idx = int((line - 100) / width)
                    if (line > 1899.9) | (line < 100):
                        flag = 1
                    else:
                        flag = 0
                else:
                    if flag == 0:
                        if construction_id == 3:
                            construction[0][matrix_idx] = construction[0][matrix_idx] + line
                        else:
                            construction[construction_id + 1][matrix_idx] = construction[construction_id + 1][
                                                                                matrix_idx] + line

        for construction_id, chargeMZ in enumerate(theoryvec):
            # m/z by charge is not exist
            if len(chargeMZ) == 0:
                continue
            for line_id, line in enumerate(chargeMZ):
                if line_id % 2 == 0:
                    matrix_idx = int((line - 100) / width)
                    if (line > 1899.9) | (line < 100):
                        flag = 1
                    else:
                        flag = 0
                else:
                    if flag == 0:
                        construction[construction_id + 4][matrix_idx] = construction[construction_id + 4][
                                                                            matrix_idx] + line

        construction = np.asarray(construction, dtype=float)
        transformer = Normalizer()
        construction = transformer.fit_transform(construction)
        X = torch.FloatTensor([construction])
        feature = torch.FloatTensor(addFeatvec)
        y = self.Y[idx]
        weight = self.weight[idx]
        return X, y, feature, weight



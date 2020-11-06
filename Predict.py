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
import matplotlib.pyplot as plt
import os
import sys
from DataUtils import DefineTestDataset
from DataUtils import readTestData
from model import CNN
from model import Net
from model import mymodel

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def test_model(model, test_data, device):
    model.eval()
    test_loader = Data.DataLoader(test_data, batch_size=64,num_workers=8,pin_memory=True)

    y_true, y_pred, y_pred_prob = [], [], []
    for data, feature in test_loader:
        data, feature = Variable(data), Variable(feature)
        data, feature = data.to(device,non_blocking=True), feature.to(device,non_blocking=True)

        output = model(data, feature)
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        pred_prob = torch.softmax(output.data, dim=1).cpu()
        pred_prob = np.asarray(pred_prob, dtype=float)
        y_pred.extend(pred)
        y_pred_prob.extend(pred_prob[:, 1].tolist())
    return y_pred_prob


def main(iexp,itheory,ifeature,output,fmodel):
    start = time.time()
    L, idset = readTestData(iexp,itheory,ifeature)
    L_idx = [i for i in range(len(L))]
    test_data = DefineTestDataset(L_idx, L)
    device = torch.device("cuda")
    model = mymodel(CNN(), Net())
    model.cuda()
    model = nn.DataParallel(model)
    model.to(device)
    #model.load_state_dict(torch.load('./temp_model_second/epoch148.pt', map_location=lambda storage, loc: storage))
    model.load_state_dict(torch.load(fmodel, map_location=lambda storage, loc: storage))
    y_pred = test_model(model, test_data, device)
    print(time.time() - start)

    # f = open('./marine2/OSU_D2_FASP_Elite_02262014_' + str(i) + '.pin')
    # fw=open('./marine2/OSU_rerank_'+str(i)+'.csv','w')
    # fw_test = open('prob' + str(i) + '.txt', 'w')
    fw_test = open(output, 'w')
    count=0
    for line_id, line in enumerate(idset):
        # if line_id == 0:
        # fw.write('rerank_score' + ',' + line)
        # continue
        # fw.write(str(y_pred[line_id - 1]) + ',' + line)
        if line == False:
            fw_test.write(str(line) + ':' + 'None' + '\n')
        else:
            fw_test.write(line + ':' + str(y_pred[count]) + '\n')
            count+=1
    # f.close()
    # fw.close()
    fw_test.close()


if __name__ == "__main__":
    inputfile_exp=sys.argv[1]
    inputfile_theory=sys.argv[2]
    inputfile_feature=sys.argv[3]
    outputfile=sys.argv[4]
    modelfile=sys.argv[5]
    main(inputfile_exp,inputfile_theory,inputfile_feature,outputfile,modelfile)

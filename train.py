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
from DataUtils import DefineDataset
from DataUtils import readData
from model import CNN
from model import Net
from model import mymodel
from model import my_loss

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(data, model, loss, device):
    # Evaluation, return accuracy and loss

    model.eval()  # set mode to evaluation to disable dropout
    data_loader = Data.DataLoader(data,batch_size=6)

    data_len = len(data)
    total_loss = 0.0
    y_true, y_pred = [], []

    for data, label, feature, weight in data_loader:
        data, label, feature, weight = Variable(data), Variable(label), Variable(feature), Variable(weight)
        data, label, feature, weight = data.to(device), label.to(device), feature.to(device), weight.to(device)

        output = model(data, feature)
        losses = loss(output, label, weight)

        total_loss += losses.data.item()
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(label.data)

    acc = (np.array(y_true) == np.array(y_pred)).sum()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    Pos_prec = 0
    Neg_prec = 0

    if y_pred.count(1) == 0:
        Pos_prec = 0
    elif y_pred.count(0) == 0:
        Neg_prec = 0
    else:
        for idx in range(len(y_pred)):
            if y_pred[idx] == 1:
                if y_true[idx] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if y_true[idx] == 1:
                    TN += 1
                else:
                    FN += 1

        Pos_prec = TP / (TP + FP)
        Neg_prec = FN / (TN + FN)

    return acc / data_len, total_loss / data_len, Pos_prec, Neg_prec


def test_model(model, test_data, device):
    print("Testing...")
    model.eval()
    start_time = time.time()
    test_loader = Data.DataLoader(test_data,batch_size=6)

    model.load_state_dict(torch.load(
        'cnn_pytorch.pt', map_location=lambda storage, loc: storage))

    y_true, y_pred, y_pred_prob = [], [], []
    for data, label, feature, weight in test_loader:
        y_true.extend(label.data)
        data, label, feature, weight = Variable(data), Variable(label), Variable(feature), Variable(weight)
        data, label, feature, weight = data.to(device), label.to(device), feature.to(device), weight.to(device)

        output = model(data, feature)
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        pred_prob = torch.softmax(output.data, dim=1).cpu()
        pred_prob = np.asarray(pred_prob, dtype=float)
        y_pred.extend(pred)
        y_pred_prob.extend(pred_prob[:, 1].tolist())

    test_acc = metrics.accuracy_score(y_true, y_pred)
    test_f1 = metrics.f1_score(y_true, y_pred, average='macro')
    print(
        "Test accuracy: {0:>7.2%}, F1-Score: {1:>7.2%}".format(test_acc, test_f1))

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(
        y_true, y_pred, target_names=['T', 'D']))

    print('Confusion Matrix...')
    cm = metrics.confusion_matrix(y_true, y_pred)
    print(cm)

    print("Time usage:", get_time_dif(start_time))

def train_model(x_train, x_test, L, Y, weight):
    LR = 1e-4
    start_time = time.time()
    train_data = DefineDataset(x_train, L, Y, weight)
    test_data = DefineDataset(x_test, L, Y, weight)
    device = torch.device("cuda")
    model = mymodel(CNN(), Net())
    model.cuda()
    model = nn.DataParallel(model)
    model.to(device)
    # criterion = nn.CrossEntropyLoss(size_average=False)
    #model.load_state_dict(torch.load('./temp_model/epoch54.pt', map_location=lambda storage, loc: storage))
    criterion = my_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best_acc = 0.0
    Train_acc = []
    Test_acc = []
    for epoch in range(0, 150):
        # load the training data in batch
        batch_count = 0
        batch_time = time.time()
        model.train()
        train_loader = Data.DataLoader(train_data,batch_size=6)
        start = time.time()
        for x_batch, y_batch, feature, weight in train_loader:
            end = time.time()
            batch_count = batch_count + 1
            inputs, targets, feature, weight = Variable(x_batch), Variable(y_batch), Variable(feature), Variable(weight)
            inputs, targets, feature, weight = inputs.to(device), targets.to(device), feature.to(device), weight.to(
                device)
            optimizer.zero_grad()
            outputs = model(inputs, feature)  # forward computation
            loss = criterion(outputs, targets, weight)
            # backward propagation and update parameters
            loss.backward()
            optimizer.step()
            # print("batch"+str(batch_count)+" :"+str(get_time_dif(batch_time)))

            # evaluate on both training and test dataset

        train_acc, train_loss, train_Posprec, train_Negprec = evaluate(train_data, model, criterion, device)
        test_acc, test_loss, test_PosPrec, test_Negprec = evaluate(test_data, model, criterion, device)
        if test_acc > best_acc:
            # store the best result
            best_acc = test_acc
            torch.save(model.state_dict(), 'benchmark.pt')
        name = './temp_model/epoch' + str(epoch) + '.pt'
        torch.save(model.state_dict(), name)
        time_dif = get_time_dif(start_time)
        msg = "Epoch {0:3}, Train_loss: {1:>7.2}, Train_acc {2:>6.2%}, Train_Posprec {3:>6.2%}, Train_Negprec {" \
              "4:>6.2%}, " + "Test_loss: {5:>6.2}, Test_acc {6:>6.2%},Test_Posprec {7:6.2%}, Test_Negprec {8:6.2%} " \
                             "Time: {9} "
        print(msg.format(epoch + 1, train_loss, train_acc, train_Posprec, train_Negprec, test_loss, test_acc,
                         test_PosPrec, test_Negprec, time_dif))
        Train_acc.append(train_acc)
        Test_acc.append(test_acc)
    # torch.save(model.state_dict(), 'cnn_pytorch.pt')
    test_model(model, test_data, device)
    return Test_acc, Train_acc

if __name__ == "__main__":
    expPrefix = sys.argv[1]
    theoryPrefix = sys.argv[2]
    featurePrefix = sys.argv[3]
    LabelPrefix = sys.argv[4]
    filenum = sys.argv[5]
    start = time.time()
    L, Y, weight = readData(expPrefix, theoryPrefix, featurePrefix, LabelPrefix, filenum)
    print(len(L))
    end = time.time()
    print(end - start)
    L_idx = [i for i in range(len(L))]
    # L_idx=random.sample(L_idx,len(L_idx))
    # X_train, X_test, y_train, y_test = train_test_split(L_idx,L_idx , test_size=0.1, random_state=10)
    X_train, X_test, y_train, y_test = train_test_split(L_idx, Y, test_size=0.1, random_state=10)
    train_model(X_train, X_test, L, Y, weight)

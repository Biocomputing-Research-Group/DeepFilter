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


def readData(expfix, theoreticfix, featurefix, Labelfix, filenum):
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


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.1)


class my_loss(torch.nn.Module):
    def __init__(self):
        super(my_loss, self).__init__()

    def forward(self, outputs, targets, weight_label):
        weight_label = weight_label.float()
        entropy = -F.log_softmax(outputs, dim=1)
        w_entropy = weight_label * entropy[:, 1] + (1 - weight_label) * entropy[:, 0]
        losssum = torch.sum(w_entropy)
        return losssum


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 7), dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        self.conv1.apply(init_weights)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 5)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3))
        )
        self.conv2.apply(init_weights)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 6)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3))

        )
        self.conv3.apply(init_weights)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2, 6)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3))

        )
        self.conv4.apply(init_weights)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(3072, 1024)
        # self.fc2 = nn.Linear(2048,1680)
        self.fc3 = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x=self.fc2(x)
        return F.relu(self.fc3(x))
        # return F.softmax(self.fc3(x), dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(11, 512)

    def forward(self, x):
        return F.relu(self.layer1(x))


class mymodel(nn.Module):
    def __init__(self, CNN, Net):
        super(mymodel, self).__init__()
        self.CNN = CNN
        self.Net = Net
        self.fc = nn.Linear(1024, 2)

    def forward(self, x1, x2):
        x1 = self.CNN(x1)
        x2 = self.Net(x2)
        x = torch.cat((x1, x2), dim=1)
        return self.fc(x)


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(data, model, loss, device):
    # Evaluation, return accuracy and loss

    model.eval()  # set mode to evaluation to disable dropout
    data_loader = Data.DataLoader(data,batch_size=256)

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
    test_loader = Data.DataLoader(test_data,batch_size=256)

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


def performance_plot(testacc, acc, nb_epochs):
    testacc = testacc
    acc = acc
    nb_epochs = nb_epochs
    x = []
    for i in range(nb_epochs):
        x.append(i + 1)

    plt.figure()
    plt.plot(x, testacc, 'b', label='test dataset')
    plt.plot(x, acc, 'r', label='training dataset')
    for a, b in zip(x, testacc):
        plt.text(a, b, "(%.0f,%.3f)" %
                 (a, b), ha='center', va='bottom', fontsize=7)
    for a, b in zip(x, acc):
        plt.text(a, b, "(%.0f,%.3f)" %
                 (a, b), ha='center', va='bottom', fontsize=7)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('CNN_PSM_Performance')
    plt.legend(loc='upper right')
    plt.show()


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
    model.load_state_dict(torch.load('./temp_model/epoch54.pt', map_location=lambda storage, loc: storage))
    criterion = my_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best_acc = 0.0
    Train_acc = []
    Test_acc = []
    for epoch in range(54, 111):
        # load the training data in batch
        batch_count = 0
        batch_time = time.time()
        model.train()
        train_loader = Data.DataLoader(train_data,batch_size=256)
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
            torch.save(model.state_dict(), 'cnn_pytorch.pt')
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
    print('done')

import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import random

from torch.utils.data import TensorDataset, DataLoader


def open_excel(filename):
    """
    打开数据集，进行数据处理
    :param filename:文件名
    :return:特征集数据、标签集数据
    """
    readbook = pd.read_excel(f'{filename}.xlsx', engine='openpyxl')
    nplist = readbook.T.to_numpy()
    data = nplist[9:17].T
    data = np.float64(data)
    target = nplist[18:26].T
    target = np.float64(target)
    return data, target

def random_number(data_size, key):
    """
   使用shuffle()打乱
    """
    number_set = []
    for i in range(data_size):
        number_set.append(i)

    if key == 1:
        random.shuffle(number_set)

    return number_set

def split_data_set(data_set, target_set, rate, ifsuf):
    """
    说明：分割数据集，默认数据集的rate是测试集
    :param data_set: 数据集
    :param target_set: 标签集
    :param rate: 测试集所占的比率
    :return: 返回训练集数据、测试集数据、训练集标签、测试集标签
    """
    # 计算训练集的数据个数
    train_size = int((1 - rate) * len(data_set))
    # 随机获得数据的下标
    data_index = random_number(len(data_set), ifsuf)
    # 分割数据集（X表示数据，y表示标签），以返回的index为下标
    # 训练集数据
    x_train = data_set[data_index[:train_size]]
    # 测试集数据
    x_test = data_set[data_index[train_size:]]
    # 训练集标签
    y_train = target_set[data_index[:train_size]]
    # 测试集标签
    y_test = target_set[data_index[train_size:]]

    return x_train, x_test, y_train, y_test

def inputtotensor(inputtensor, labeltensor):
    """
    将数据集的输入和标签转为tensor格式
    :param inputtensor: 数据集输入
    :param labeltensor: 数据集标签
    :return: 输入tensor，标签tensor
    """
    inputtensor = np.array(inputtensor)
    inputtensor = torch.FloatTensor(inputtensor)

    labeltensor = np.array(labeltensor)
    labeltensor = torch.FloatTensor(labeltensor)

    return inputtensor, labeltensor

def addbatch(data_train, data_test, batchsize):
    """
    设置batch
    :param data_train: 输入
    :param data_test: 标签
    :param batchsize: 一个batch大小
    :return: 设置好batch的数据集
    """
    data = TensorDataset(data_train, data_test)
    data_loader = DataLoader(data, batch_size=batchsize, shuffle=False)

    return data_loader

def train_test(traininput, trainlabel, testinput, testlabel, batchsize):
    """
    函数输入为：训练输入，训练标签，测试输入，测试标签，一个batch大小
    进行BP的训练，每训练一次就算一次准确率，同时记录loss
    :return:训练次数list，训练loss，测试loss，准确率
    """
    # 设置batch
    traindata = addbatch(traininput, trainlabel, batchsize)  # shuffle打乱数据集
    min_loss = 0.4
    for epoch in range(1000000000):
        for step, data in enumerate(traindata):
            net.train()
            inputs, labels = data
            # 前向传播
            out = net(inputs)
            # 计算损失函数
            loss = loss_func(out, labels)
            loss.clone().detach().requires_grad_(True)
            # 清空上一轮的梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
        # 测试准确率
        #if epoch % 100 == 0:
        net.eval()
        testout = net(testinput)
        testloss = loss_func(testout, testlabel)
        if testloss.item()< min_loss:
            min_loss = testloss.item()
            # 保存
            print("saving")
            torch.save(net.state_dict(),"CNN_917.1875.pt")
        print("训练次数为", epoch, "的loss为:", testloss.item())

#main
data, target=open_excel('917.1875')
print("done readin")
# 数据划分为训练集和测试集和是否打乱数据集
split = 0.3  # 测试集占数据集整体的多少
ifshuffle = 0  # 1为打乱数据集，0为不打乱

x_train, x_test, y_train, y_test = split_data_set(data, target, split, ifshuffle)

# 将数据转为tensor格式
traininput, trainlabel = inputtotensor(x_train, y_train)
testinput, testlabel = inputtotensor(x_test, y_test)

# 归一化处理
traininput = nn.functional.normalize(traininput)
testinput = nn.functional.normalize(testinput)
# trainlabel = nn.functional.normalize(trainlabel)
# testlabel = nn.functional.normalize(testlabel)

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)  # 1为in_channels 10为out_channels
        self.conv2 = torch.nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)  # 1为in_channels 10为out_channels
        self.conv4 = torch.nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(1)
        self.fc = torch.nn.Linear(8, 8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn1(x) #输出=（输入-均值）/（标准差+eps),衔接上下层结构，装逼用
        x = self.conv3(x) #con3,4用于提高复杂度，更能提取特征
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn2(x)
        #x = x.view(batchsize,-1)
        x1 = x #残差，可去除
        x = self.fc(x)
        x = x1 + x #残差，可去除
        #x = self.relu(x)
        return x

# 创建神经网络实例和损失函数
net = Net()
loss_func = nn.MSELoss()
optimizer = optim.Adam(params=net.parameters(), lr=0.001, eps=1e-8)
batchsize=20
traininput=traininput.reshape(-1, 1, 100, 8) #(batch, channel, row, col)
testinput=testinput.reshape(-1, 1, 100, 8)
trainlabel=trainlabel.reshape(-1, 1, 100, 8)
testlabel=testlabel.reshape(-1, 1, 100, 8)
# 训练并且记录每次准确率，loss     函数输入为：训练输入，训练标签，测试输入，测试标签，一个batch大小
train_test(traininput, trainlabel, testinput, testlabel, batchsize)

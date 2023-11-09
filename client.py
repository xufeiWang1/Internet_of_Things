#客户端
#调用load
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from load import *  #   load.py 里定义了函数getdata，用于边缘端数据划分，all_labels_train和all_features_train就是要用的
from torch import optim  # 包括常用的优化算法如SGD,Adam，Adagrad等
from Model import tianqi_2NN

# import numpy as np
# import pandas as pd
# from sklearn import  preprocessing
# import warnings
args = {
    'num_of_clients': 10,
    'num_comn': 10,
    'batchsize': 32,
    'epoch': 5,
    'learning_rate': 0.001,
    'input_size': 13,
    'hidden_size': 5,
    'output_size': 1
}

class ClientsGroup(object):  # 构造边缘端集合类
    def __init__(self, dev, class_num, Net):
        self.dev = dev
        self.clients_set = set()
        self.class_num = class_num
        self.Net = Net
        self.fc1_weight = torch.from_numpy(np.zeros((args['hidden_size'], args['input_size']))).float()
        self.fc1_bias = torch.from_numpy(np.zeros((args['hidden_size'],))).float()
        self.fc2_weight = torch.from_numpy(np.zeros((args['output_size'], args['hidden_size']))).float()
        self.fc2_bias = torch.from_numpy(np.zeros((args['output_size'],))).float()
        for i in range(self.class_num):
            Client = client()
            Client.clientNet = tianqi_2NN(args['input_size'], args['hidden_size'], args['output_size'])
            self.clients_set.add(Client)


    def dataSetBalanceAllocation(self):  # 初始化集合的内容
        trainData = getdata()  # getdata是load.py里的函数
        num_rows = trainData.shape[0]  # =3000  # 获取数据的行数
        shuffled_indices = np.random.permutation(num_rows)          # 打乱行索引的顺序
        shuffled_array = trainData[shuffled_indices]          # 根据打乱后的行索引重新排列数组
        reshaped_array = shuffled_array.reshape(10, 300, 14)           # 重新变成 10 个 300×14 维的数组

        for i, client in enumerate(self.clients_set):
            client.train_ds = reshaped_array[i, :, :13]
            client.target_data = reshaped_array[i, :, 13:]
            client.dev = self.dev
            client.num_example = client.train_ds.shape[0]

    def updateSet(self, lossFun):
        for client in self.clients_set:
            localBatchSize = client.train_ds.shape[0]
            localepoch = 5
            client.localUpdate(localBatchSize, localepoch, lossFun)

    def combineParameters(self):
        for client in self.clients_set:
            self.fc1_weight = self.fc1_weight + client.fc1_weight
            self.fc1_bias = self.fc1_bias + client.fc1_bias
            self.fc2_weight = self.fc2_weight + client.fc2_weight
            self.fc2_bias = self.fc2_bias + client.fc2_bias
        self.fc1_weight = self.fc1_weight / 10
        self.fc1_bias = self.fc1_bias / 10
        self.fc2_weight = self.fc2_weight / 10
        self.fc2_bias = self.fc2_bias / 10

    def send_parameter(self):
        for client in self.clients_set:
            client.receiveParameters(self.fc1_weight, self.fc1_bias, self.fc2_weight, self.fc2_bias)

class client(object):  # 构造每个边缘类
    def __init__(self, train_ds=None, target_data=None, dev=None, num_example=None, clientNet=None):
        self.train_ds = train_ds
        self.target_data = target_data
        self.dev = dev
        self.clientNet = clientNet
        self.fc1_weight = None
        self.fc1_bias = None
        self.fc2_weight = None
        self.fc2_bias = None
        self.train_dl = None
        self.num_example = num_example
        self.state = {}

    def localUpdate(self, localBatchSize, localepoch, lossFun):  # 本地计算函数
        opti = optim.SGD(self.clientNet.parameters(), lr=args['learning_rate'])  #  不能去掉
        for epoch in range(localepoch):
            localparameters = list(self.clientNet.parameters())
            # 前向传播
            output = self.clientNet(self.train_ds)  # 会出问题么没用到localBatchSize
            #  计算损失
            loss = lossFun(output, torch.from_numpy(self.target_data).float())
            # 反向传播和参数更新
            opti.zero_grad()
            loss.backward()
            opti.step()

            # 打印损失
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

        # 获取参数
        parameters = list(self.clientNet.parameters())
        # 获取第一个线性层的权重和偏置参数
        self.fc1_weight = parameters[0].data
        self.fc1_bias = parameters[1].data

        # 获取第二个线性层的权重和偏置参数
        self.fc2_weight = parameters[2].data
        self.fc2_bias = parameters[3].data

    def receiveParameters(self, fc1_weight, fc1_bias, fc2_weight, fc2_bias):
        params = {
            'fc1.weight': fc1_weight,
            'fc1.bias': fc1_bias,
            'fc2.weight': fc2_weight,
            'fc2.bias': fc2_bias
        }

        # 将参数传输给新的神经网络
        self.clientNet.load_state_dict(params)
        self.fc1_weight = fc1_weight
        self.fc1_bias = fc1_bias
        self.fc2_weight = fc2_weight
        self.fc2_bias = fc2_bias

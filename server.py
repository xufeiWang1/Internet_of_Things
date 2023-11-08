# 参数服务器
# 调用client、load、model
# import numpy as np
# import pandas as pd
# from sklearn import  preprocessing
# import warnings

import os
import argparse
#  设定全局的参数——客户端数量、全局轮数、每个边缘端训练轮数localepoch、batchsize、learning rate等
import torch
import torch.cuda
from tqdm import tqdm  # 用于显示进度条
from torch import optim  # 包括常用的优化算法如SGD,Adam，Adagrad等
# from client import  ClientsGroup  #
import sys
from Model import tianqi_2NN
from load import *





class ClientsGroup(object):  # 构造边缘端集合类
    def __init__(self, dev, class_num):
        self.dev = dev
        self.clients_set = set()
        self.class_num = class_num
        self.dataSetBalanceAllocation()

    def creat_clients_net(self):
        client1_net = tianqi_2NN('input_size', 'hidden_size', 'output_size')
        client2_net = tianqi_2NN('input_size', 'hidden_size', 'output_size')
        client3_net = tianqi_2NN('input_size', 'hidden_size', 'output_size')
        client4_net = tianqi_2NN('input_size', 'hidden_size', 'output_size')
        client5_net = tianqi_2NN('input_size', 'hidden_size', 'output_size')
        client6_net = tianqi_2NN('input_size', 'hidden_size', 'output_size')
        client7_net = tianqi_2NN('input_size', 'hidden_size', 'output_size')
        client8_net = tianqi_2NN('input_size', 'hidden_size', 'output_size')
        client9_net = tianqi_2NN('input_size', 'hidden_size', 'output_size')
        client10_net = tianqi_2NN('input_size', 'hidden_size', 'output_size')

    def add_client(self, client):
        self.clients_set.add(client)


    def client_create(self):
        client1 = client()
        client2 = client()
        client3 = client()
        client4 = client()
        client5 = client()
        client6 = client()
        client7 = client()
        client8 = client()
        client9 = client()
        client10 = client()
        self.add_client(client1)
        self.add_client(client2)
        self.add_client(client3)
        self.add_client(client4)
        self.add_client(client5)
        self.add_client(client6)
        self.add_client(client7)
        self.add_client(client8)
        self.add_client(client9)
        self.add_client(client10)

    def dataSetBalanceAllocation(self):  # 初始化集合的内容
        index_class = getdata(self.class_num)  # getdata是load.py里的函数
        # 获取数据的行数
        num_rows = index_class.shape[0]  # =3000
        # 打乱行索引的顺序
        shuffled_indices = np.random.permutation(num_rows)
        # 根据打乱后的行索引重新排列数组
        shuffled_array = index_class[shuffled_indices]
        # 重新变成 10 个 300×14 维的数组
        reshaped_array = shuffled_array.reshape(10, 300, 14)

        self.client_create()

        for client in range(self.clients_set):
            client.train_ds = reshaped_array[i, :, :13]
            client.target_data = reshaped_array[i, :, 13:]
            client.dev = self.dev
            client.num_example = client.train_ds.shape[0]
            client.clientNet = tianqi_2NN('input_size', 'hidden_size', 'output_size')

    def updateSet(self, Net, lossFun, opti):
        for client in range(self.clients_set):
            localBatchSize = client.train_ds.shape[0]
            localepoch = 5
            client.localUpdate(localBatchSize, localepoch, lossFun, opti)

            # someone_1 = client(TensorDataset(torch.tensor(local_data, dtype=torch.float, requires_grad = True), torch.tensor.....))

            # self.clients_set['client{}'.format(i)] = someone_1


class client(object):  # 构造每个边缘类
    def __init__(self, train_ds=None, target_data=None, dev=None, num_example=None, clientNet = None):
        self.train_ds = train_ds
        self.target_data = target_data
        self.dev = dev
        self.clientNet = clientNet
        self.train_dl = None
        self.num_example = num_example
        self.state = {}

    def localUpdate(self, localBatchSize, localepoch, lossFun, opti):  # 本地计算函数
        parameters = self.clientNet.parameters()
        for epoch in range(localepoch):
            # 前向传播
            output = self.clientNet(self.train_ds)  #  会出问题么没用到localBatchSize
            # 计算损失
            loss = lossFun(output, self.target_data)
            # 反向传播和参数更新
            opti.zero_grad()
            loss.backward()
            opti.step()

            # 打印损失
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

        # 获取更新后的参数值
        updated_parameters = []
        for param in parameters:
            updated_param = param.data.clone()  # 克隆参数值
            updated_parameters.append(updated_param)  # updated_parameters 列表中存储了更新后的参数值
        return updated_parameters


#   设置全局参数
args = {
    'num_of_clients': 10,
    'num_comn': 10,
    'batchsize': 32,
    'epoch': 5,
    'learning_rate': 0.001,
    'input_size': 10,
    'hidden_size': 5,
    'output_size': 1
}

# clients_in_comn_100 = ['client1', 'client2', 'client3', 'client4', 'client5',
#                        'client6', 'client7', 'client8', 'client9', 'client10',
#                        ]  #

net = tianqi_2NN('input_size', 'hidden_size', 'output_size')  # 实例化神经网络
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = torch.nn.DataParallel(net)

if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")

net = net.to(dev)

loss_func = torch.nn.MSELoss(reduction='mean')  # 确定损失函数和优化器
opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

myClients = ClientsGroup(dev, args['num_of_clients'])  # 实例化边缘端集合对象

for i in range(1, args['num_comn'] + 1):  # 边缘端计算
    print('-------------------------fedavg----------------------')
    print('------------------------------第', i, '次训练--------------------')
    sum_parameters = None
    for client in tqdm(clients_in_comn_100):
        print(client)
        local_parameters = myClients.clients_set[client].localUpdate(args['batchsize'], args['epoch'], net, loss_func,
                                                                     opti, global_parameters)  # 本地参数

        for var in sum_parameters:  # 本地模型聚合
            sum_parameters[var] = sum_parameters[var] + local_parameters[var] * example
    for var in global_parameters:
        global_parameters[var] = sum_parameters[var] / sum_example

x = torch.tensor(all_features_test, dtype=torch.float)  # 验证拟合和预测结果
x = x.to(dev)  # 将输入数据和目标数据移动到设备上
predict_1 = net(x)  # __call__()方法像函数一样调用对象
predict = predict_1.cpu().detach().numpy()

#参数服务器
#调用client、load、model
# import numpy as np
# import pandas as pd
# from sklearn import  preprocessing
# import warnings

import os
import argparse
#  设定全局的参数——客户端数量、全局轮数、每个边缘端训练轮数localepoch、batchsize、learning rate等
import torch
import torch.cuda
from tqdm import  tqdm   #  用于显示进度条
from torch import optim  #  包括常用的优化算法如SGD,Adam，Adagrad等
from client import  ClientsGroup  #
import sys
from Model import tianqi_2NN
from load import *

#   设置全局参数
args = {
    'num_of_clients': 10,
    'num_comn': 10,
    'batchsize': 32,
    'epoch': 5,
    'learning_rate': 0.001
}

input_size = 10
hidden_size = 5
output_size = 1

clients_in_comn_100 = ['client1', 'client2', 'client3', 'client4', 'client5',
                       'client6', 'client7', 'client8', 'client9', 'client10',
                       ]  #

net = tianqi_2NN(input_size, hidden_size, output_size)  #  实例化神经网络
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = torch.nn.DataParallel(net)

if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")

net = net.to(dev)


loss_func = torch.nn.MSELoss(reduction='mean')  #  确定损失函数和优化器
opti = optim.SGD(net.parameters(), lr = args['learning_rate'])


myClients = ClientsGroup(dev, args['num_of_clients'])  #  实例化边缘端集合对象

for i in range(1, args['num_comn']+1):  #  边缘端计算
    print('-------------------------fedavg----------------------')
    print('------------------------------第', i, '次训练--------------------')
    sum_parameters = None
    for client in tqdm(clients_in_comn_100):
        print(client)
        local_parameters = myClients.clients_set[client].localUpdate(args['batchsize'], args['epoch'], net, loss_func, opti, global_parameters) #本地参数


        for var in sum_parameters :  #  本地模型聚合
            sum_parameters[var] = sum_parameters[var] + local_parameters[var] * example
    for var in global_parameters:
        global_parameters[var] = sum_parameters[var] / sum_example


x = torch.tensor(all_features_test, dtype = torch.float)  #  验证拟合和预测结果
x = x.to(dev)  #  将输入数据和目标数据移动到设备上
predict_1 = net(x)  #  __call__()方法像函数一样调用对象
predict = predict_1.cpu().detach().numpy()






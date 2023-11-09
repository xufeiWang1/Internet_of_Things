import os
import argparse
import torch
import torch.cuda
from tqdm import tqdm  # 用于显示进度条
from torch import optim  # 包括常用的优化算法如SGD,Adam，Adagrad等
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
    'learning_rate': 0.001,
    'input_size': 13,
    'hidden_size': 5,
    'output_size': 1
}

if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")

myClients = ClientsGroup(dev, args['num_of_clients'])  # 实例化边缘端集合对象

if torch.cuda.device_count() >= 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    myClients.Net = torch.nn.DataParallel(myClients.Net)

myClients.Net = myClients.Net.to(dev)


# myClients.client_create()
for i in range(1, args['num_comn'] + 1):  # 边缘端计算
    print('-------------------------fedavg----------------------')
    print('------------------------------第', i, '次训练--------------------')
    myClients.dataSetBalanceAllocation()
    myClients.updateSet()
    myClients.combineParameters()


test_data = getTestData()
x = torch.tensor(test_data, dtype=torch.float)  # 验证拟合和预测结果
x = x.to(dev)  # 将输入数据和目标数据移动到设备上
t =x[:, :13]
predict_1 = myClients.Net(x[:, :13])  # __call__()方法像函数一样调用对象
predict = predict_1.cpu().detach().numpy()
loss_test = myClients.loss_func(predict_1, x[:, 13])
print(f" Loss: {loss_test.item()}")

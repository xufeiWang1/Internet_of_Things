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
    'learning_rate': 0.01,
    'input_size': 13,
    'hidden_size': 5,
    'output_size': 1
}

class ClientsGroup(object):  # 构造边缘端集合类
    def __init__(self, dev, class_num):
        self.dev = dev
        self.clients_set = set()
        self.class_num = class_num
        self.Net = tianqi_2NN(args['input_size'], args['hidden_size'], args['output_size'])
        self.loss_func = torch.nn.MSELoss(reduction='mean')  # 确定损失函数和优化器
        self.opti = optim.SGD(self.Net.parameters(), lr=args['learning_rate'])

        for i in range(self.class_num):
            Client = client(self.dev)
            Client.localNet = tianqi_2NN(args['input_size'], args['hidden_size'], args['output_size'])
            Client.opti = optim.SGD(Client.localNet.parameters(), lr=args['learning_rate'])
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

    def updateSet(self):
        for client in self.clients_set:
            localBatchSize = client.train_ds.shape[0]
            localepoch = 5
            client.localUpdate(localepoch)

    def combineParameters(self):
        for i, clie in enumerate(self.clients_set):
            localparameters = list(clie.localNet.parameters())
            if i == 0 :
                localparameters_dict = {'local_fc1_weight': localparameters[0].data, 'local_fc1_bias': localparameters[1].data,
                                        'local_fc2_weight': localparameters[2].data, 'local_fc2_bias': localparameters[3].data}
            else :
                localparameters_dict['local_fc1_weight'] = localparameters_dict['local_fc1_weight']+ localparameters[0].data
                localparameters_dict['local_fc1_bias'] = localparameters_dict['local_fc1_bias']+ localparameters[1].data
                localparameters_dict['local_fc2_weight'] = localparameters_dict['local_fc2_weight']+ localparameters[2].data
                localparameters_dict['local_fc2_bias'] = localparameters_dict['local_fc2_bias']+ localparameters[3].data
        localparameters_dict['local_fc1_weight'] = localparameters_dict['local_fc1_weight']/ 10
        localparameters_dict['local_fc1_bias'] = localparameters_dict['local_fc1_bias']/ 10
        localparameters_dict['local_fc2_weight'] = localparameters_dict['local_fc2_weight']/ 10
        localparameters_dict['local_fc2_bias'] = localparameters_dict['local_fc2_bias']/ 10
        params = {
            'module.fc1.weight': localparameters_dict['local_fc1_weight'],
            'module.fc1.bias': localparameters_dict['local_fc1_bias'],
            'module.fc2.weight': localparameters_dict['local_fc2_weight'],
            'module.fc2.bias': localparameters_dict['local_fc2_bias']
        }
        self.Net.load_state_dict(params)
        params = {
            'fc1.weight': localparameters_dict['local_fc1_weight'],
            'fc1.bias': localparameters_dict['local_fc1_bias'],
            'fc2.weight': localparameters_dict['local_fc2_weight'],
            'fc2.bias': localparameters_dict['local_fc2_bias']
        }
        for clie in self.clients_set:
            clie.localNet.load_state_dict(params)


class client(object):  # 构造每个边缘类
    def __init__(self,  dev):
        self.train_ds = None
        self.target_data = None
        self.dev = dev
        self.localNet = None
        self.num_example = None
        self.locallossfun = torch.nn.MSELoss(reduction='mean')  # 确定损失函数和优化器
        self.opti = None

    def localUpdate(self, localepoch):  # 本地计算函数
        for epoch in range(localepoch):
            # 前向传播
            output = self.localNet(self.train_ds)  # 会出问题么没用到localBatchSize
            #  计算损失
            loss = self.locallossfun(output, torch.from_numpy(self.target_data).float())
            # 反向传播和参数更新
            self.opti.zero_grad()
            loss.backward()
            self.opti.step()

            # 打印损失
            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")




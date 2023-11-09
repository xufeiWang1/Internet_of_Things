#神经网络
import numpy as np
import pandas as pd
from sklearn import  preprocessing
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class tianqi_2NN(nn.Module):  #  构建神经网络，继承自nn.Module类
    def __init__(self, input_size, hidden_size, output_size):
        super(tianqi_2NN, self).__init__()  # 调用父类的构造函数
        self.fc1 = nn.Linear(input_size, hidden_size)  #创造两个全连接层对象，将它们分配给fc1和fc2
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self,inputs):
        if isinstance(inputs, np.ndarray):
            tensor = F.sigmoid(self.fc1(torch.from_numpy(inputs).float()))
        elif isinstance(inputs, torch.Tensor):
            tensor = F.sigmoid(self.fc1(inputs))
        tensor = self.fc2(tensor)
        return tensor


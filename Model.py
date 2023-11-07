#神经网络
import numpy as np
import pandas as pd
from sklearn import  preprocessing
import warnings

import torch.nn as nn
import torch.nn.functional as F

in_feat_1 = 7
out_feat_1 = 50
in_feat_2 = out_feat_1
out_feat_2 = 1

class tianqi_2NN(nn.Module):  #  构建神经网络，继承自nn.Module类
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        self.fc1 = nn.Linear(in_feat_1, out_feat_1)  #创造两个全连接层对象，将它们分配给fc1和fc2
        self.fc2 = nn.Linear(in_feat_2, out_feat_2)

    def forward(self,inputs):
        tensor = F.sigmoid(self.fc1(inputs))
        tensor = self.fc2(tensor)
        return tensor
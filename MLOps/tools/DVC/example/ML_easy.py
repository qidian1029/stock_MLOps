import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

df_data  = pd.read_csv('C:/Users/is_li/Desktop/paper/github/stock on MLOps/MLOps/tools/DVC/example/data/dataset.csv',
index_col= [0,1],header=[0,1])

df_data = df_data.fillna(0)

import torch
import torch.nn as nn
import torch.nn.functional	
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 创建两个列表，用来存储数据的特征和标签
data_feat, data_target = [],[]

# 设每条数据序列有20组数据
seq = 10
feature_num = 10

for index in range(len(df_data) - seq):
    # 构建特征集
    data_feat.append(df_data['feature'].iloc[:,range(feature_num)][index: index + seq].values)
    # 构建target集
    data_target.append(df_data['label'][index:index + seq])

# 将特征集和标签集整理成numpy数组
data_feat = np.array(data_feat)
data_target = np.array(data_target)

# 这里按照8:2的比例划分训练集和测试集
test_set_size = int(np.round(0.1*df_data.shape[0]))  # np.round(1)是四舍五入，
train_size = data_feat.shape[0] - (test_set_size) 

trainX = torch.from_numpy(data_feat[:train_size].reshape(-1,seq,feature_num)).type(torch.Tensor)   
testX  = torch.from_numpy(data_feat[train_size:].reshape(-1,seq,feature_num)).type(torch.Tensor)
trainY = torch.from_numpy(data_target[:train_size].reshape(-1,seq,1)).type(torch.Tensor)
testY  = torch.from_numpy(data_target[train_size:].reshape(-1,seq,1)).type(torch.Tensor)

batch_size=5
train = torch.utils.data.TensorDataset(trainX,trainY)
test = torch.utils.data.TensorDataset(testX,testY)
train_loader = torch.utils.data.DataLoader(dataset=train, 
                                           batch_size=batch_size, 
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test, 
                                          batch_size=batch_size, 
                                          shuffle=False)

import torch.nn as nn
input_dim = feature_num    # 数据的特征数
hidden_dim = 2    # 隐藏层的神经元个数
num_layers = 2     # LSTM的层数
output_dim = 1     # 预测值的特征数
                   #（这是预测股票价格，所以这里特征数是1，如果预测一个单词，那么这里是one-hot向量的编码长度）
class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)  # 把形状改回来
        return x

# 定义模型
num_epochs = 50
model = LSTM(input_size=input_dim, hidden_size=hidden_dim, output_size=output_dim, num_layers=num_layers)

for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())
    
loss_function = nn.MSELoss()  # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # 优化器


train_loss = [] 
for epoch in range(num_epochs):
    out = model(trainX)
    loss = loss_function(out, trainY)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    train_loss.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print('Epoch: {}, Loss:{:.10f}'.format(epoch + 1, loss.item()))

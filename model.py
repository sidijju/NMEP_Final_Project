import torch
import torch.nn as nn
import numpy as np

class Simple(nn.Module):

    def __init__(self, num_classes=7):
        super(Simple, self).__init__()
        self.embedding = nn.Embedding(1000, 32)
        self.lstm = nn.LSTM(32, 32, 10, batch_first=True) # try 200*32 if doesn't work
        self.dense1 = nn.Linear(32, 16)
        self.dense2 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(0.7)


    def forward(self, x):
        #print(x.shape)
        x = self.embedding(x)
        #print(x.shape)
        #x = x.permute(1, 0, 2)
        #print(x.shape)
        x, hidden = self.lstm(x) # 512 x 200 x 32
        #x = x.contiguous().view(-1, 32)
        #print(x.shape)
        x = x[:, -1, :]
        #print(x.shape)

        #print("hidden" + str(hidden.shape))
        x = self.dense1(x)
        x = self.dropout(x)
        #print(x.shape)
        x = nn.functional.relu(x)
        #print(x.shape)
        x = self.dense2(x)
        #print(x.shape)
        return x
        #return nn.functional.softmax(x, dim=1)

class RNN_LSTM(nn.Module):

    def __init__(self, num_classes=7):
        super(RNN_LSTM, self).__init__()
        self.embedding = nn.Embedding(1000, 32)


    def forward(self, x):
        return x

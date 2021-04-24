import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    
    def __init__(self, num_classes=7):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(1000, 32)
        self.lstm = nn.LSTM(32, 32, 1) # try 200*32 if doesn't work
        self.dense1 = nn.Linear(32, 16)
        self.dense2 = nn.Linear(16, num_classes)


    def forward(self, x):
        #x = torch.from_numpy(np.array(x)).float()
        #x = torch.stack(x)
        x = self.embedding(x)
        x = self.lstm(x)
        x = nn.functional.relu(self.dense1(x))
        x = self.dense2(x)
        return nn.functional.softmax(x)

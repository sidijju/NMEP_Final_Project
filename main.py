import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import shutil
import yaml
import argparse
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torchvision import transforms
from torchsummary import summary
from torch.utils.data.dataset import Dataset

import model as Model
import data

parser = argparse.ArgumentParser(description='Configuration details for training/testing genre classifier')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--train', action='store_true')
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--image', type=str)
parser.add_argument('--model_number', type=str, required=True)

args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    correct = 0
    total = 0
    for i, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(input)
        predicted = torch.argmax(outputs, 1)
        loss = criterion(outputs, target)

        for index in range(len(input)):
            if predicted[index] == target[index]:
                correct += 1
            total += 1

        #if i % 100 == 0:
        #    print("loss: ", loss.item())
        #    print("label: ", target[0].item())
        #    print("predicted: ", predicted[0].item())

        loss.backward()
        optimizer.step()

    return loss.item(), correct/total

def validate(val_loader, model, criterion):
    model.eval()
    correct = 0
    total = 0
    for i, (input, target) in enumerate(val_loader):
        outputs = model(input)
        predicted = torch.argmax(outputs, 1)
        loss = criterion(outputs, target)
        for index in range(len(input)):
            if predicted[index] == target[index]:
                correct += 1
            total += 1

        #if i % 100 == 0:
        #    print("loss: ", loss.item())
        #    print("label", target[0].item())
        #    print("predicted: ", predicted[0].item())

    return loss.item(), correct/total

def save_checkpoint(state, best_one, filename='genre_checkpoint.pth.tar', filename2='genre_best.pth.tar'):
    torch.save(state, filename)
    #best_one stores whether your current checkpoint is better than the previous checkpoint
    if best_one:
        shutil.copyfile(filename, filename2)

n_epochs = config["num_epochs"]
model =
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=config["learning_rate"],
                      momentum=config["momentum"],
                      weight_decay=config["weight_decay"])
train_dataset =
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_dataset =
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)

best_loss = 1
for epoch in range(n_epochs):
     train_loss, train_acc = train(train_loader, model, criterion, optimizer, config["num_epochs"])
     val_loss, val_acc = validate(val_loader, model, criterion)
 	 #TODO: Save your checkpoint
     best_one = val_loss < best_loss
     save_checkpoint(model.state_dict(), best_one)
     print("Epoch %d - Train Accuracy: %2.4f, Validation Accuracy: %2.4f" % (epoch, train_acc, val_acc))

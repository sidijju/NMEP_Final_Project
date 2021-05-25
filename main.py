import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#import time
import shutil
import yaml
import argparse
import matplotlib.pyplot as plt

#from torch.autograd import Variable
#from torchvision import transforms
from torchsummary import summary
from torch.utils.data.dataset import Dataset

from model import Simple
from data import DataTrain, DataTest

config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    correct = 0
    total = 0
    for i, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(input)
        logits = nn.functional.softmax(outputs, dim=1)
        predicted = torch.argmax(logits, 1)
        loss = criterion(outputs, target)

        for index in range(len(input)):
            #print(predicted[index])
            #print(target[index])
            if predicted[index] == target[index]:
                correct += 1
            total += 1

        if i % 100 == 0:
            print("loss: ", loss.item())
            print("label: ", target[0].item())
            print("predicted: ", predicted[0].item())
            print("outputs:", logits[0])

        loss.backward()
        optimizer.step()

    return loss.item(), correct/total

def validate(val_loader, model, criterion):
    model.eval()
    correct = 0
    total = 0
    for i, (input, target) in enumerate(val_loader):
        outputs = model(input)
        logits = nn.functional.softmax(outputs, dim=1)
        predicted = torch.argmax(logits, 1)
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
    if best_one:
        shutil.copyfile(filename, filename2)

n_epochs = config["num_epochs"]
training_size = config["training_size"]
num_classes = config["num_classes"]
model = Simple(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=config["learning_rate"],
                      momentum=config["momentum"],
                      weight_decay=config["weight_decay"])

train_dataset = DataTrain(training_size)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_dataset = DataTest(training_size)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)

best_loss = 1
for epoch in range(n_epochs):
     train_loss, train_acc = train(train_loader, model, criterion, optimizer, config["num_epochs"])
     val_loss, val_acc = validate(val_loader, model, criterion)
 	 # Save checkpoint
     best_one = val_loss < best_loss
     save_checkpoint(model.state_dict(), best_one)
     print("Epoch %d - Train Accuracy: %2.4f, Validation Accuracy: %2.4f" % (epoch + 1, train_acc, val_acc))

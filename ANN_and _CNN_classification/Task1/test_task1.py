# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import datasets

import time

#%matplotlib inline

np.random.seed(0)
torch.manual_seed(0)
data = np.load('data_task1.npy') #load data
label=np.load('label_task1.npy') #load label

data_train, data_test, label_train, label_test = train_test_split(data,label,
    test_size=0.25, random_state=75)
    
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=20, out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=36)
        self.fc3 = nn.Linear(in_features=36, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=28)
        self.fc5 = nn.Linear(in_features=28, out_features=24)
        self.fc6 = nn.Linear(in_features=24, out_features=12)
        #self.fc7 = nn.Linear(in_features=18, out_features=12)
        self.output = nn.Linear(in_features=12, out_features=3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))        
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        #x = F.relu(self.fc7(x))
        x = self.output(x)
        return x
        
data_train = torch.FloatTensor(data_train)
data_test = torch.FloatTensor(data_test)
label_train = torch.LongTensor(label_train)
label_test = torch.LongTensor(label_test)

model = ANN()
model

start_time = time.time()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#epochs = 1000
epochs = 800
loss_arr = []
train_loss = []
train_accuracy = []

for i in range(epochs):
    y_hat = model.forward(data_train)
    loss = criterion(y_hat, label_train)

    accuracy = len(np.where(label_train == y_hat.argmax(1))[0]) / len(label_train)
    train_accuracy.append(accuracy)
    train_loss.append(loss.item())
 
    if i % 10 == 0:
        print(f'Epoch: {i} Loss: {loss}')
 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
fig, ax = plt.subplots(2, 1, figsize=(12,8))
ax[0].plot(train_loss)
ax[0].set_ylabel('Loss')
ax[0].set_title('Training Loss')

ax[1].plot(train_accuracy)
ax[1].set_ylabel('Classification Accuracy')
ax[1].set_title('Training Accuracy')

plt.tight_layout()
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))

y_hat_test = model(data_test)
test_accuracy= len(np.where(label_test == y_hat_test.argmax(1))[0]) / len(label_test)
print("Test Accuracy {:.2f}".format(test_accuracy))

# Specify a path
PATH = "task1model.pt"

# Save
torch.save(model, PATH)

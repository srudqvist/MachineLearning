#!/usr/bin/env python3
# ols.py                                                     SSimmons March 2018
"""
Uses a neural net to find the ordinary least-squares regression model. Trains
with batch gradient descent, and computes r^2 to gauge predictive quality.
"""

import torch
import pandas as pd
import torch.nn as nn
import time

# Read the named columns from the csv file into a dataframe.
names = ['SalePrice','1st_Flr_SF','2nd_Flr_SF','Lot_Area','Overall_Qual',
    'Overall_Cond','Year_Built','Year_Remod/Add','BsmtFin_SF_1','Total_Bsmt_SF',
    'Gr_Liv_Area','TotRms_AbvGrd','Bsmt_Unf_SF','Full_Bath']
df = pd.read_csv('Project2/Assignment5/AmesHousing.csv', names = names)
data = df.values # read data into a numpy array (as a list of lists)
data = data[1:] # remove the first list which consists of the labels
data = data.astype(float) # coerce the entries in the numpy array to floats
data = torch.FloatTensor(data) # convert data to a Torch tensor

data.sub_(data.mean(0)) # mean-center
data.div_(data.std(0))  # normalize

xss = data[:,1:]
yss = data[:,:1]

# define a model class
class MyModel(nn.Module):

  def __init__(self):
    super(MyModel, self).__init__()
    #self.layer1 = nn.Linear(13, 1)

    #self.layer1 = nn.Linear(13, 10)
    #self.layer2 = nn.Linear(10, 1)

    self.layer1 = nn.Linear(13, 10)
    self.layer2 = nn.Linear(10, 8)
    self.layer3 = nn.Linear(8, 1)

    # self.layer1 = nn.Linear(13, 10)
    # self.layer2 = nn.Linear(10, 8)
    # self.layer3 = nn.Linear(8, 4)
    # self.layer4 = nn.Linear(4, 1)

  def forward(self, xss):
    #return self.layer1(xss)
    # xss = self.layer1(xss)
    # xss = torch.relu(xss)
    # return self.layer2(xss)

    xss = self.layer1(xss)
    xss = torch.relu(xss)
    xss = self.layer2(xss)
    xss = torch.relu(xss)
    return self.layer3(xss)

    # xss = self.layer1(xss)
    # xss = torch.relu(xss)
    # xss = self.layer2(xss)
    # xss = torch.relu(xss)
    # xss = self.layer3(xss)
    # xss = torch.relu(xss)
    # return self.layer4(xss)

# create and print an instance of the model class
model = MyModel()
print(model)

# This is used to implement momentum (by hand) -- specifically, it remembers the
# values of the weights from the previous iteration during the training loop.
z_parameters = []
for param in model.parameters():
  z_parameters.append(param.data.clone())
for param in z_parameters:
  param.zero_()



criterion = nn.MSELoss()

num_examples = len(data)
batch_size = 20         # 2264  10  5
learning_rate = .008    # 0.000355  0.001
momentum = 0.9          # 0.899
epochs = 2000           # 1000

start_time = time.time()
# train the model
for epoch in range(epochs):
  for _ in range(num_examples//batch_size):
    # randomly pick batchsize examples from data
    indices = torch.randperm(num_examples)[:batch_size]

    yss_mb = yss[indices]  # the targets for the mb (minibatch)
    yhatss_mb = model(xss[indices])  # model outputs for the mb

    loss = criterion(yhatss_mb, yss_mb)
    model.zero_grad()
    loss.backward() # back-propagate

    # update weights
    for i, (z_param, param) in enumerate(zip(z_parameters, model.parameters())):
      z_parameters[i] = momentum * z_param + param.grad.data
      param.data.sub_(z_parameters[i] * learning_rate)
    # for f in model.parameters():
    #   f.data.sub_(f.grad.data * learning_rate * momentum)

  with torch.no_grad():
    total_loss = criterion(model(xss), yss).item()
  if epoch in range(0,10):
    print('epoch: {0}, loss: {1:11.8f}'.format(epoch+1, total_loss))
  if epoch == 10:
    print("...")
  if epoch in range(epochs -10, epochs):
    print('epoch: {0}, loss: {1:11.8f}'.format(epoch+1, total_loss))

end_time = time.time()
tot_time = end_time - start_time

print("total number of examples:", num_examples, end='')
print("; batch size:", batch_size)
print("learning rate:", learning_rate)
print("momentum:", momentum)
if tot_time < 60:
  print(f"time: {tot_time}")
else:
  print(f"time: {tot_time / 60}min")

# Compute 1-SSE/SST which is the proportion of the variance in the data
# explained by the regression hyperplane.
SS_E = 0.0;  SS_T = 0.0
mean = data.mean(0)[0] # mean of the outputs (zero, if data are mean-centered)
for datum in data:
  SS_E = SS_E + (datum[0] - model(datum[1:]))**2
  SS_T = SS_T + (datum[0] - mean)**2
print('1-SSE/SST = {:1.4f}'.format(1.0-(SS_E/SS_T).item()))
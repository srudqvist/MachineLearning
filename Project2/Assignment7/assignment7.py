# Machine Learning 2022
# Assignment 7
# Samuel Rudqvist

import torch
import pandas as pd
import torch.nn as nn
import du.lib as dulib
import time
import random

# Read the named columns from the csv file into a dataframe.
names = ['SalePrice','1st_Flr_SF','2nd_Flr_SF','Lot_Area','Overall_Qual',
    'Overall_Cond','Year_Built','Year_Remod/Add','BsmtFin_SF_1','Total_Bsmt_SF',
    'Gr_Liv_Area','TotRms_AbvGrd','Bsmt_Unf_SF','Full_Bath']
df = pd.read_csv('Project2/Assignment5/AmesHousing.csv', names = names)
data = df.values # read data into a numpy array (as a list of lists)
data = data[1:] # remove the first list which consists of the labels
# print(data)

# randomize the order of the data
random.shuffle(data)
#print("after")
#print(data)


data = data.astype(float) # coerce the entries in the numpy array to floats
data = torch.FloatTensor(data) # convert data to a Torch tensor


#data = data.random_(data)

# data.sub_(data.mean(0)) # mean-center
# data.div_(data.std(0))  # normalize

xss = data[:,1:]
yss = data[:,:1]

xss_train = xss[:2000]
xss_test = xss[2000:]

xss_train.sub_(xss_train.mean(0))
xss_train.div_(xss_train.std(0))
xss_test.sub_(xss_test.mean(0))
xss_test.div_(xss_test.std(0))

yss_train = yss[:2000]
yss_test = yss[2000:]

yss_train.sub_(yss_train.mean(0))
yss_train.div_(yss_train.std(0))
yss_test.sub_(yss_test.mean(0))
yss_test.div_(yss_test.std(0))

print(xss)
print(len(xss_train))
print(len(xss_test))

# define a model class
class MyModel(nn.Module):

  def __init__(self):
    super(MyModel, self).__init__()

    self.layer1 = nn.Linear(13, 10)
    self.layer2 = nn.Linear(10, 8)
    self.layer3 = nn.Linear(8, 1)

  def forward(self, xss):

    xss = self.layer1(xss)
    xss = torch.relu(xss)
    xss = self.layer2(xss)
    xss = torch.relu(xss)
    return self.layer3(xss)

# create and print an instance of the model class
model = MyModel()
print(model)

criterion = nn.MSELoss()

# num_examples = len(data)
# batch_size = 20         # 2264  10  5
# learning_rate = .008    # 0.000355  0.001
# momentum = 0.9          # 0.899
# epochs = 2000           # 1000

start_time = time.time()
# train the model
model = dulib.train(
  model,
  criterion,
  train_data = (xss_train, yss_train),
  #test_data = (xss_test, yss_test),
  learn_params = {'lr': 0.008, 'mo': 0.9},
  epochs = 2000
)

# get the weights of the trained model
# params = list(model.parameters())
# print(params)
# m = params[0].item(); b = params[1].item()

end_time = time.time()
tot_time = end_time - start_time

# print("total number of examples:", num_examples, end='')
# print("; batch size:", batch_size)
# print("learning rate:", learning_rate)
# print("momentum:", momentum)
if tot_time < 60:
  print(f"time: {tot_time}")
else:
  print(f"time: {tot_time / 60}min")

print("explained variation:", dulib.explained_var(model, (xss_train, yss_train)))
print("test data variation:", dulib.explained_var(model, (xss_test, yss_test)))
print()

# Compute 1-SSE/SST which is the proportion of the variance in the data
# explained by the regression hyperplane.
# SS_E = 0.0;  SS_T = 0.0
# mean = data.mean(0)[0] # mean of the outputs (zero, if data are mean-centered)
# for datum in data:
#   SS_E = SS_E + (datum[0] - model(datum[1:]))**2
#   SS_T = SS_T + (datum[0] - mean)**2
# print('1-SSE/SST = {:1.4f}'.format(1.0-(SS_E/SS_T).item()))
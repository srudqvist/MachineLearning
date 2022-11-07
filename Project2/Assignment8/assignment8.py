# Machine Learning 2022
# Assignment 7
# Samuel Rudqvist

import copy
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

# randomize the order of the data
random.shuffle(data)

# data = data[torch.randperm(len(data))]

data = data.astype(float) # coerce the entries in the numpy array to floats
data = torch.FloatTensor(data) # convert data to a Torch tensor

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

    self.layer1 = nn.Linear(20, 12)
    self.layer2 = nn.Linear(12, 8)
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

epochs = 200
learning_rate = 0.008
momentum = 0.9
batchsize = 20

folds = 10
bail_after = 30
no_improvement = 0
best_valids = 1e15*torch.ones(folds)

start_time = time.time()

while no_improvement < bail_after:

  model, valids = dulib.cross_validate(
      k = folds,
      model = model,
      crit = criterion,
      train_data = (xss_train[:,1:], yss_train[:,:1]),
      cent_norm_feats = (True, True),
      cent_norm_targs = (True, True),
      epochs = epochs,
      learn_params = {'lr':learning_rate, 'mo':momentum},
      bs = batchsize,
      verb = 10
  )

  if valids.mean().item() < best_valids.mean().item():
    best_model = copy.deepcopy(model)
    best_valids = valids
    no_improvement = 0
  else:
    no_improvement += 1

# train the model

# model = dulib.train(
#   model,
#   criterion,
#   train_data = (xss_train, yss_train),
#   #test_data = (xss_test, yss_test),
#   learn_params = {'lr': 0.008, 'mo': 0.9},
#   epochs = 2000,
#   graph = True
#   #batchsize = 20
# )


end_time = time.time()
tot_time = end_time - start_time

if tot_time < 60:
  print(f"time: {tot_time}")
else:
  print(f"time: {tot_time / 60}min")

print("training data variation:", dulib.explained_var(model, (xss_train, yss_train)))
print("test data variation:", dulib.explained_var(model, (xss_test, yss_test)))
print()

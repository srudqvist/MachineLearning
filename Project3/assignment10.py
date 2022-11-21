



import torch
import pandas as pd
import torch.nn as nn
import du.lib as dulib
import time
import random
from skimage import io


# read in all of the digits
digits = io.imread('Project3/digits.png')
xss = torch.Tensor(5000,400)
idx = 0
for i in range(0, 1000, 20):
  for j in range(0, 2000, 20):
    xss[idx] = torch.Tensor((digits[i:i+20,j:j+20]).flatten())
    idx = idx + 1

# extract just the zeros and eights from xss
tempxss = torch.Tensor(1000,400)
tempxss[:500] = xss[:500]
tempxss[500:] = xss[4000:4500]

# overwrite the original xss with just zeros and eights
xss = tempxss

# generate yss to hold the correct classification for each example
yss = torch.Tensor(len(xss),1)
for i in range(len(yss)):
  yss[i] = i//500


# xss.sub_(xss.mean(0)) # mean-center
# xss.div_(xss.std(0))  # normalize
# yss.sub_(yss.mean(0)) # mean-center
# yss.div_(yss.std(0))  # normalize

dulib.center(xss)
dulib.normalize(xss)


class SigmoidModel(nn.Module):

  def __init__(self):
    super(SigmoidModel, self).__init__()
    self.layer1 = nn.Linear(400, 1)

  def forward(self, x):
    x = self.layer1(x)
    return torch.sigmoid(x)

model = SigmoidModel() # create an instance of the model class

criterion = nn.MSELoss() # create an instance of the PyTorch class nn.MSELoss


model = dulib.train(
  model,
  criterion,
  train_data = (xss, yss),
  learn_params = {'lr': 0.00001, 'mo': 0.99},
  epochs = 100,
  bs = 20,
)

zero = torch.min(yss).item()
eight = torch.max(yss).item()
th = 1e-3  # threshold
cutoff = (zero+eight)/2

count = 0
for i in range(len(xss)):
  yhat = model(xss[i]).item()
  y = yss[i].item()
  if (yhat>cutoff and abs(y-eight)<th) or (yhat<cutoff and abs(y-zero)<th):
    count += 1
print("Percentage correct:",100*count/len(xss))
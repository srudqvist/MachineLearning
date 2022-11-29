



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
# tempxss = torch.Tensor(1000,400)
# tempxss[:500] = xss[:500]
# tempxss[500:] = xss[4000:4500]

# # overwrite the original xss with just zeros and eights
# xss = tempxss

# generate yss to hold the correct classification for each example
yss = torch.LongTensor(len(xss))
for i in range(len(yss)):
  yss[i] = i//500



# xss.sub_(xss.mean(0)) # mean-center
# xss.div_(xss.std(0))  # normalize
# yss.sub_(yss.mean(0)) # mean-center
# yss.div_(yss.std(0))  # normalize

# xss, xss_centered = dulib.center(xss)
# xss, xss_normalized = dulib.normalize(xss)

indices = torch.randperm(len(xss))
xss = xss[indices]; yss = yss[indices] # coherently randomize the data
xss_train = xss[:4000]; yss_train = yss[:4000]
xss_test = xss[4000:]; yss_test = yss[4000:]

xss_train, xss_train_centered = dulib.center(xss_train)
xss_test, xss_test_centered = dulib.center(xss_test)



class LogSoftMaxModel(nn.Module):

  def __init__(self):
    super(LogSoftMaxModel, self).__init__()
    self.layer1 = nn.Linear(400, 200)
    self.layer2 = nn.Linear(200, 10)

  def forward(self, xss):
    xss = self.layer1(xss)
    xss = torch.relu(xss)
    xss = self.layer2(xss)
    return torch.log_softmax(xss, dim=1)
    # return torch.sigmoid(x)

model = LogSoftMaxModel() # create an instance of the model class

criterion = nn.NLLLoss() # create an instance of the PyTorch class nn.MSELoss

model = dulib.train(
  model,
  criterion,
  train_data = (xss_train, yss_train),
  valid_data = (xss_test, yss_test),
  learn_params = {'lr': 0.0000215, 'mo': 0.99},
  epochs = 40,
  bs = 20,
  # graph = 1
)

# zero = torch.min(yss).item()
# eight = torch.max(yss).item()
# th = 1e-3  # threshold
# cutoff = (zero+eight)/2

# count = 0
# for i in range(len(xss)):
#   yhat = model(xss[i]).item()
#   y = yss[i].item()
#   if (yhat>cutoff and abs(y-eight)<th) or (yhat<cutoff and abs(y-zero)<th):
#     count += 1
# print("Percentage correct:",100*count/len(xss))

pct_training = dulib.class_accuracy(model, (xss_train, yss_train), show_cm=False)
print(f"Percentage correct on training data: {100*pct_training:.2f}")

pct_test = dulib.class_accuracy(model, (xss_test, yss_test), show_cm=False)
print(f"Percentage correct on test data: {100*pct_test:.2f}")
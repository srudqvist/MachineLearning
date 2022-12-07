import torch
import torch.nn as nn
import du.lib as dulib
from skimage import io

# read in all of the digits
digits = io.imread('digits.png')
xss = torch.Tensor(5000,400)
idx = 0
for i in range(0, 1000, 20):
  for j in range(0, 2000, 20):
    xss[idx] = torch.Tensor((digits[i:i+20,j:j+20]).flatten())
    idx = idx + 1

# generate yss to hold the correct classification for each example
yss = torch.LongTensor(len(xss))
for i in range(len(yss)):
  yss[i] = i//500

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

pct_training = dulib.class_accuracy(model, (xss_train, yss_train), show_cm=False)
print(f"Percentage correct on training data: {100*pct_training:.2f}")

pct_test = dulib.class_accuracy(model, (xss_test, yss_test), show_cm=False)
print(f"Percentage correct on test data: {100*pct_test:.2f}")
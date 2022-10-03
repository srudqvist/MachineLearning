#!/usr/bin/env python3
# ols_nn.py                                                   SSimmons Oct. 2018
"""
Uses the nn library along with auto-differentiation to define and train, using
batch gradient descent, a neural net that finds the ordinary least-squares
regression plane. The data are mean centered and normalized.  The plane obtained
via gradient descent is compared with the one obtained by solving the normal
equations.
"""
import csv
import torch
import torch.nn as nn

# Read in the data.
with open('Project1/Assignments/Assignment4/temp_co2_data.csv') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  next(csvfile)  # skip the first line of csvfile
  xss, yss = [], []
  for row in reader:
    xss.append([float(row[2]), float(row[3])])
    yss.append([float(row[1])])

# The tensors xss and yss now containing the features (i.e., inputs) and targets
# (outputs) of the data.
xss, yss = torch.tensor(xss), torch.tensor(yss)

# For validation, compute the least-squares regression plane using linear alg.
A = torch.cat((torch.ones(len(xss),1), xss), 1)
lin_alg_sol = torch.linalg.lstsq(A, yss, driver='gels').solution[:,0]

# Compute the column-wise means and standard deviations.
# Comment out the next 2 lines, for example, to see what happens if you do not
# mean center; or comment out all 4 lines; or just the last 2.
xss_means = xss.mean(0)  # xss_means.size() returns torch.size([2])
yss_means = yss.mean()   # yss_means.size() returns torch.size([])
xss_stds  = xss.std(0)   # similarly here
yss_stds  = yss.std()    # and here

# Mean center the inputs and output (if xss_means and yss_means are defined).
if 'xss_means' in locals() and 'yss_means' in locals():
  xss, yss = xss - xss_means, yss - yss_means

# Normalize the inputs and output (if xss_stds and yss_stds are defined).
if 'xss_stds' in locals() and 'yss_stds' in locals():
  xss, yss = xss/xss_stds, yss/yss_stds

# Build the model
class LinearRegressionModel(nn.Module):

  def __init__(self):
    super(LinearRegressionModel, self).__init__()
    self.layer = nn.Linear(2,1)

  def forward(self, xss):
    return self.layer(xss)

# Create an instance of the above class.
model = LinearRegressionModel()
print("The model is:\n", model)

# Set the criterion to be mean-squared error
criterion = nn.MSELoss()

learning_rate = 0.2
epochs = 30
num_examples = len(xss)
batch_size = 4

for epoch in range(epochs):  # train the model
    tot_loss = 0
    indices = torch.randperm(32)
    for index in range(num_examples):
        # yss_pred refers to the outputs predicted by the model
        yss_pred = model(xss[indices[index]] / batch_size) 

        loss = criterion(yss_pred, yss[indices[index]] / batch_size) # compute the loss
        tot_loss += loss.item() # Compute the loss

        model.zero_grad() # set the gradient to the zero vector
        loss.backward() # compute the gradient of the loss function w/r to the weights

        #adjust the weights
        for param in model.parameters():
            param.data.sub_(param.grad.data * learning_rate) 
            #param.grad(tot_loss * batch_size / num_examples)

    print("epoch: {0}, current loss: {1}".format(epoch+1, loss.item()))

# extract the weights and bias into a list
params = list(model.parameters())

# Un-mean-center and un-normalize the weights (for final validation against
# the OLS regression plane found above using linear algebra).
if not ('xss_means' in locals() and 'yss_means' in locals()):
  xss_means = torch.Tensor([0.,0.]); yss_means = 0.0
  print('no centering, ', end='')
else:
  print('centered, ', end='')
if not ('xss_stds' in locals() and 'yss_stds' in locals()):
  xss_stds = torch.Tensor([1.,1.]); yss_stds = 1.0;
  print('no normalization, ', end='')
else:
  print('normalized, ', end='')
w = torch.zeros(3)
w[1:] = params[0] * yss_stds / xss_stds
w[0] = params[1].data.item() * yss_stds + yss_means - w[1:] @ xss_means

print("The least-squares regression plane:")
# Print out the equation of the plane found by the neural net.
print("  found by the neural net is: "+"y = {0:.3f} + {1:.3f}*x1 + {2:.3f}*x2"\
    .format(w[0],w[1],w[2]))

# Print out the eq. of the plane found using closed-form linear alg. solution.
print("  using linear algebra:       y = "+"{0:.3f} + {1:.3f}*x1 + {2:.3f}*x2"\
    .format(lin_alg_sol[0], lin_alg_sol[1], lin_alg_sol[2]))
print(f"learning rate: {learning_rate}")
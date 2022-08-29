# Project0Exercises.py
# Author: Samuel Rudqvist
# Date: 08/29/2022
import torch
# -----------------------
#       Exercise 1
# -----------------------
# Create an 8 by 5 tensor initialized randomly with floats chosen from normal distributions with mean, 
# in turn, equal to 1, 2,...,40, and standard deviation equal to 4. 
# Hint: see Random Sampling; specifically, torch.normal().
# https://pytorch.org/docs/master/generated/torch.normal.html#torch.normal

test = torch.normal(mean=torch.arange(1.,41.),std=0).view(8,5) # is this correct?
example1 = torch.normal(mean=torch.arange(1.,41.),std=4).view(8,5) # is this correct?
print(test)
print(example1)
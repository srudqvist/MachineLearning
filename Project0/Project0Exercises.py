# Project0Exercises.py
# Author: Samuel Rudqvist
# Date: 08/29/2022
import torch
import matplotlib.pyplot as plt
# -----------------------
#       Exercise 1
# -----------------------
# Create an 8 by 5 tensor initialized randomly with floats chosen from normal distributions with mean, 
# in turn, equal to 1, 2,...,40, and standard deviation equal to 4. 
# Hint: see Random Sampling; specifically, torch.normal().
# https://pytorch.org/docs/master/generated/torch.normal.html#torch.normal

def example1():
    test = torch.normal(mean=torch.arange(1.,41.),std=0).view(8,5)
    example1 = torch.normal(mean=torch.arange(1.,41.),std=4).view(8,5) # is this correct?
    
    print(test)
    print(example1)


# -----------------------
#       Exercise 2
# -----------------------
# Define two tensors, each of size 40: one called xs containing the input values, 
# and a second one called ys holding the corresponding outputs. 
# Select the inputs from the uniform distribution on [0,100) and then, for each such x-value, 
# get its corresponding y-value by adding to the quantity 2x+9 a perturbation (or error) 
# randomly selected from the normal distribution with mean 0 and standard deviation 20.0. 
def example2():
    xs = torch.Tensor(40).uniform_(0,100)
    #ys = torch.Tensor(xs).multiply(2).add(9)
    
    ys = torch.Tensor(xs).multiply(-2).add(-9).normal_(mean=0,std=20.0)
    #ys = torch.Tensor(xs).multiply(2).add(9).normal_(mean=0,std=20.0)
    
    
    #ys = torch.Tensor(xs).normal_(mean=0,std=20.0).multiply(2).add(9)
    #ys = torch.Tensor(2 * xs + 9).normal_(mean=0,std=20.0)
    print(xs)
    print(ys)
    plt.scatter(xs.numpy(),ys.numpy())
    plt.show()



if __name__ == "__main__":
    #example1()
    example2()
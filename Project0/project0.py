# project0.py
# Author: Samuel Rudqvist
# Date: 08/29/2022

# Imports
import torch

def test():
    print("\nTorch Test\n")
    print(torch.Tensor(3,4).random_(3,21))


#print("File one __name__ is set to: {}" .format(__name__))
if __name__ == "__main__":
    test()
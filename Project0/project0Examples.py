# project0Examples.py
# Author: Samuel Rudqvist
# Date: 08/29/2022
# Tensor Basics

# Imports
import torch


# -----------------------------------
#           Introduction
# -----------------------------------
def intro():
    print("\nTorch Intro\n")

    # Tensors are multidimensional arrays

    # Create a 3x4 tensor initialized with numbers sampled uniformly from the set {3,4...20}
    # 4. means 4.0
    # A tensor is of type float
    # 2-dimensional tensor
    my_tensor = torch.Tensor(3,4).random_(3,21)

    # A long tensor is of type int
    my_long_tensor = torch.LongTensor(3,4).random_(3,21)

    # 3-dimensional tensor of shape 2x3x4, each entry is 7
    three_dimensional_tensor = 7*torch.ones(2,3,4)

    # Tensor with arbitrary specified entries
    arb_tensor = torch.Tensor([1,2,3,4,5,6,7,8,7,6,5,4,3,2,1])

    # Change the matrix to have a view of 5x3
    arb_tensor2 = arb_tensor.view(5,3)

    # Tensor with even numbers from 2-48, using list comprehension
    even_tensor = torch.Tensor([2*(i+1) for i in range(24)]).view(2,3,4)

    print(f"Tensor:\n{my_tensor}\n\nLong Tensor:\n{my_long_tensor}\n")
    print(f"3-dimensional tensor:\n{three_dimensional_tensor}\n")
    print(f"Arbitrary Tensor:\n{arb_tensor}\n")
    print(f"Arbitrary Tensor2:\n{arb_tensor2}\n")
    print(f"Even Tensor:\n{even_tensor}\n")

# -----------------------------------
# Dimensions, Squeezing & Unsqueezing
# -----------------------------------
def dim_squeezing():
    # torch.Tensor([2*(i+1) for i in range(24)]) returns a tensor of size 24 (not 24x1)
    # A tensor can be coerced from 4 to 4x1 with the unsqueeze method
    my_tensor = torch.Tensor([1,2,3,4]) # size 4
    print(f"\nTensor with size {my_tensor.size()}\n")

    # Unsqueezing with 0
    my_tensor2 = my_tensor.unsqueeze(0)
    print(f"Unsqueezed 0\n{my_tensor2.size()}\n")

    # Unsqueezing with 1
    my_tensor3 = my_tensor.unsqueeze(1)
    print(f"Unsqueezed 1\n{my_tensor3.size()}\n")

    # Dimensions of 1 can be squeezed 
    my_tensor4 = my_tensor3.squeeze(1)
    print(f"Squeezed {my_tensor4.size()}\n")

    # Get the dimension with the dim() method
    print(f"Dimension: {my_tensor.dim()}\n")

    # There are 0-dimensional tensors (scalars) (notice lowercase t in tensor)
    zero_dim_tensor = torch.tensor(17.3)
    print(f"0-dim tensor: {zero_dim_tensor.size()}\n")

    # Get the value of zero_dim_tensor with the item() method
    print(f"0-dim tensor value: {zero_dim_tensor.item()}\n")

    # Scalars and numbers
    my_tensor5 = torch.tensor([1.,2.,3.,4.,5.])
    print(f"Torch Scalar: {my_tensor5[4]}")
    print(f"Pyhton Number: {my_tensor5[4].item()}")

    # .shape and .size can be used interchangeably (notice no parenthesis on shape)
    print(f"Size: {torch.rand(2,7,10).size()}")
    print(f"Shape: {torch.rand(2,7,10).shape}")


# -----------------------------------
#  Arithmetic Operations On Tensors
# -----------------------------------
def arithmetic_operations():
    # Addition
    my_tensor = 7*torch.ones(2,3,4)
    my_tensor2 = torch.Tensor([i+1 for i in range(24)]).view(2,3,4)
    
    print(f"my_tensor = {my_tensor}")
    print(f"my_tensor2 = {my_tensor2}")
    print("\nAddition:")
    print(f"{my_tensor + my_tensor2}")

    # Multiplication
    print("\nMultiplication:")
    print(f"{my_tensor * my_tensor2}")

    # Unsqueeze if different dimensions
    #print(torch.ones(3,4) * torch.Tensor([1,2,3])) GIVES ERROR
    print(torch.ones(3,4) * torch.Tensor([1,2,3]).unsqueeze(1))
    print("\nDivision:")
    print(torch.ones(3,4) / torch.Tensor([1,2,3,4]).unsqueeze(0))
    

# -----------------------------------
#         Slicing tensors
# -----------------------------------
def slicing_tensors():
    print("\nArange:")
    print(torch.arange(1,25).type())
    print(torch.arange(1.,25.).type())

    my_tensor = torch.ones(5,2)
    print(f"\nmy_tensor:\n{my_tensor}")
    my_tensor[:,1] = torch.Tensor([0,1,2,3,4])  # Pythonic slicing, remember that Python has zero indexing
    print(f"\nmy_tensor after slicing:\n{my_tensor}")

    print("\nMore Slicing\n")
    print(my_tensor[3,1])  #  returns tensor(3.) which is the entry in the 4th row, 2nd column of x
    print(my_tensor[3,1].item())  #  returns 3.0 the actual float in the 4th row, 2nd column of x
    print(my_tensor[:,0])  # returns the first column of x
    print(my_tensor[1:-1,1])  # returns a tensor comprising the 1st through the 2nd to last elements of the 2nd column of x
    print(my_tensor[1:4,1])  # same as the last command
    print(my_tensor[4,:])  # returns the fifth row of x
    print(my_tensor[:,:])  # returns all of x (so the same as just x)
    print(torch.rand(2,3,4)[:,2,:])  # returns a 2x4 tensor
    print(torch.rand(2,7,4)[:,2:5,:])  # returns a 2x3x4 tensor


#print("File one __name__ is set to: {}" .format(__name__))
if __name__ == "__main__":
    #intro()
    #dim_squeezing()
    #arithmetic_operations()
    slicing_tensors()
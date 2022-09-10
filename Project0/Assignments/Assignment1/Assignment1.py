from turtle import st
import torch
import statistics
import time

# Question 1
# Describe the output of the following program in one or two sentences.
# Why is the output different every time you run the program?
def number_one():
    xs = torch.randn(30)
    print(xs)
    print(xs.shape)
    print(xs.size)

'''
Answer:
It prints a tensor of size 30. with pseudo random values from a normal distribution with mean 0 and 
variance 1. (the standard normal distribution)
The output is different each time since the numbers are pseudo random from the normal distribution, 
and since there is a variance of 1 it makes it even more unlikely to return the exact same tensor.
'''


# Question 2
# Add lines to the program above that print the mean and standard distribution
# Is the mean exactly 0? 
# Is the standard deviation exactly 1?
def number_two():
    xs = torch.randn(30)
    print(xs)
    print(f"Mean: {xs.mean()}")
    print(f"STD: {xs.std()}")

'''
The mean is not exactly 0
The standard deviation is not exactly 1
The mean is not exactly 0 because of the sampling from the normal distribution, the numbers are more likely
to land close to 0 for the mean.
The standard deviation is not exactly 1 since the numbers are spread out under the curve.
'''

def number_three():
    xs = torch.Tensor(30).normal_(mean=100, std=25)
    print(xs)
    print(f"Mean: {xs.mean()}")

'''

'''


def number_four():
    list_of_means_100 = []
    list_of_means_1000 = []
    list_of_means_10000 = []
    list_of_means_1000000 = []

    start_time = time.time()
    for i in range(0,1000000):
        xs = torch.Tensor(30).normal_(mean=100, std=25)
        list_of_means_1000000.append(float(xs.mean()))
        if i < 10000:
            list_of_means_10000.append(float(xs.mean()))
        if i < 100:
            list_of_means_100.append(float(xs.mean()))
        if i < 1000:
            list_of_means_1000.append(float(xs.mean()))
    
    end_time = time.time()
    avg_100 = sum(list_of_means_100)/len(list_of_means_100)
    avg_1000 = sum(list_of_means_1000)/len(list_of_means_1000)
    avg_10000 = sum(list_of_means_10000)/len(list_of_means_10000)
    avg_1000000 = sum(list_of_means_1000000)/len(list_of_means_1000000)
    
    print(f"Time: {end_time - start_time}")
    print(f"The average mean of 100 runs is: {avg_100}")
    print(f"The average mean of 1000 runs is: {avg_1000}")
    print(f"The average mean of 10000 runs is: {avg_10000}")
    print(f"The average mean of 1000000 runs is: {avg_1000000}")

'''
The more times the program is run, the closer to 100 the mean is
'''

def number_five():
    list_of_std_100 = []
    list_of_std_1000 = []
    list_of_std_10000 = []
    list_of_std_1000000 = []

    start_time = time.time()

    for i in range(0,1000000):
        xs = torch.Tensor(30).normal_(mean=100, std=25)
        list_of_std_1000000.append(float(xs.std()))
        if i < 10000:
            list_of_std_10000.append(float(xs.std()))
        if i < 100:
            list_of_std_100.append(float(xs.std()))
        if i < 1000:
            list_of_std_1000.append(float(xs.std()))

    end_time = time.time()

    avg_100 = sum(list_of_std_100)/len(list_of_std_100)
    avg_1000 = sum(list_of_std_1000)/len(list_of_std_1000)
    avg_10000 = sum(list_of_std_10000)/len(list_of_std_10000)
    avg_1000000 = sum(list_of_std_1000000)/len(list_of_std_1000000)

    print(f"Time: {end_time - start_time}")
    print(f"The average std of 100 runs is: {avg_100}")
    print(f"The average std of 1000 runs is: {avg_1000}")
    print(f"The average std of 10000 runs is: {avg_10000}")
    print(f"The average std of 1000000 runs is: {avg_1000000}")


'''
The standard deviation does not seem to be effected by times run?
'''

# What happens if you sample from the uniform distribution on [0,1] 
# instead of a normal distribution?
# What value does the mean of the means of sample size 30 appear to target?
# What about the mean of the standard deviations of many samples of size 30?
def number_six():
    # Create lists to hold the means
    list_of_means_100 = []
    list_of_means_1000 = []
    list_of_means_10000 = []
    list_of_means_1000000 = []
    list_of_std_100 = []
    list_of_std_1000 = []
    list_of_std_10000 = []
    list_of_std_1000000 = []

    xs = torch.Tensor(30).uniform_(0,1)
    print(f"Mean: {xs.mean()}\nSTD: {xs.std()}")
    # Loop to create data for the lists
    start_time = time.time()
    for i in range(0,1000000):
        xs = torch.Tensor(30).uniform_(0,1)
        list_of_means_1000000.append(float(xs.mean()))
        list_of_std_1000000.append(float(xs.std()))
        if i < 10000:
            list_of_means_10000.append(float(xs.mean()))
            list_of_std_10000.append(float(xs.std()))
        if i < 100:
            list_of_means_100.append(float(xs.mean()))
            list_of_std_100.append(float(xs.std()))
        if i < 1000:
            list_of_means_1000.append(float(xs.mean()))
            list_of_std_1000.append(float(xs.std()))

    end_time = time.time()
    # Get the mean for each
    avg_100 = sum(list_of_means_100)/len(list_of_means_100)
    avg_1000 = sum(list_of_means_1000)/len(list_of_means_1000)
    avg_10000 = sum(list_of_means_10000)/len(list_of_means_10000)
    avg_1000000 = sum(list_of_means_1000000)/len(list_of_means_1000000)

    avg_std_100 = sum(list_of_std_100)/len(list_of_std_100)
    avg_std_1000 = sum(list_of_std_1000)/len(list_of_std_1000)
    avg_std_10000 = sum(list_of_std_10000)/len(list_of_std_10000)
    avg_std_1000000 = sum(list_of_std_1000000)/len(list_of_std_1000000)
    
    print(f"Time: {end_time - start_time}")
    print(f"The mean of the mean of 100 runs is: {avg_100}")
    print(f"The mean of the mean of 1000 runs is: {avg_1000}")
    print(f"The mean of the mean of 10000 runs is: {avg_10000}")
    print(f"The mean of the mean of 1000000 runs is: {avg_1000000}")
    
    print(f"The mean of the std of 100 runs is: {avg_std_100}")
    print(f"The mean of the std of 1000 runs is: {avg_std_1000}")
    print(f"The mean of the std of 10000 runs is: {avg_std_10000}")
    print(f"The mean of the std of 1000000 runs is: {avg_std_1000000}")

'''
The more times its run, the closer to 0.5 the mean is
'''
    
    #print(f"Mean: {xs.mean()}")

# took 353s
def insane():
    list_of_means_1000000 = []

    start_time = time.time()
    for i in range(0,1000000):
        xs = torch.Tensor(30000).normal_(mean=100, std=25)
        list_of_means_1000000.append(float(xs.mean()))
    
    end_time = time.time()
    avg_1000000 = sum(list_of_means_1000000)/len(list_of_means_1000000)
    
    print(f"Time: {end_time - start_time}")
    print(f"The average mean of 1000000 runs is: {avg_1000000}")

if __name__ == '__main__':
    #number_one()
    #number_two()
    #number_three()
    #number_four()
    #number_five()
    number_six()
    #insane()
import torch
import csv
import matplotlib.pyplot as plt

with open("Project0/Assignments/Assignment2/assign2.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(csvfile) # skip the first line
    xs, ys = [], []
    for row in reader:
        xs.append(float(row[0]))
        ys.append(float(row[1]))
xs, ys = torch.tensor(xs), torch.tensor(ys)

#print(xs)

#

#ys = ys.unsqueeze(1)
#xs = xs.mm(ys)
#xs = xs.unsqueeze(0)
#xs = xs.inverse()
#xs = xs.transpose(0,1).mm(xs).inverse()


# Calculate the slope
# y = ax + b -> a = y/x - b

# Least squares regression line: w = ((((X^T)*X)^-1)*X^T)y
# Hints:

# To create $X$, first create a tensor of size 40x2 consisting of all ones. 
# Then modify the second column of $X$ so that it holds the $x$-values of our data.
# Then use the appropriate combination of the basic linear-algebraic tensor operations listed above to compute $w$.


# xs = xs.unsqueeze(0)
# # (((X^T)*X)^-1)
# w = xs.transpose(0,1).mm(xs).inverse()
# print(w)
# print()
# w = w@w


# print(xs.size())
# #slope = ys/xs

# print(w)
# #print(slope)



# Use torch.ones to put ones next to the x values
# then X = torch.column_stack((torch.ones(60), xs))
# then transpose X
# then XTXINVERSE = xtranspose.mm(x).inverse()
# then y = torch.unsqueeze(ys,1)
# XTY = (Xtranspose.mm(y))
# answer = XTXInverse.mm(XTY)
# print(answer)

X = torch.column_stack((torch.ones(60), xs))
#print(X)
X_Transpose = X.transpose(0,1)
# (XTX)^-1
xtx = X_Transpose.mm(X).inverse() 
Y = torch.unsqueeze(ys,1)
xty = X_Transpose.mm(Y)
ans = xtx.mm(xty)
printable_ans = ans[1], ans[0]
#print(ans[1], ans[0])
print(ans) # second value is slope, first value is intercept?
# plt.scatter(xs.numpy(),ys.numpy())
# plt.plot(ans.numpy())
# plt.show()


# Simmons in class
# build a design matrix
# put all ones in the first column, x values in second

#w = (X.transpose(0,1).mm(X).inverse()).mm(X.transpose(0,1).mm(Y.unsqueeze(1)))
X = torch.column_stack((torch.ones(60), xs))

w = (X.transpose(0,1).mm(X).inverse()).mm(X.transpose(0,1).mm(ys.unsqueeze(1)))
#w = (X.transpose(1,0).mm(X).inverse()).mm(X.transpose(1,0).mm(ys.unsqueeze(1)))
#print(w)

# plt.scatter(xs.numpy(),ys.numpy())
# plt.plot(w.numpy())
# plt.show()


# thomas code
# xs = xs.unsqueeze(0)

# xs2 = xs.squeeze(0)

# XS = xs @ xs2 #Dot product
# ys = ys.unsqueeze(1)

# xsys = xs.mm(ys)

# xsys.inverse()

# X = xsys.transpose(0,1).mm(xsys).inverse()

# Z = X * XS

# plt.scatter(xs.numpy(), ys.numpy())
# plt.plot(Z.numpy(), color = 'r')
# plt.show()
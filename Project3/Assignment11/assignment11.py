
import torch
import torch.nn as nn
import du.lib as dulib
from skimage import io

# read in all of the digits
digits = io.imread('Project3/digits.png')
xss = torch.Tensor(5000, 20, 20)
idx = 0
for i in range(0, 1000, 20):
    for j in range(0, 2000, 20):
        xss[idx] = torch.Tensor((digits[i:i+20, j:j+20]))
        idx = idx + 1

# generate yss to hold the correct classification for each example
yss = torch.LongTensor(len(xss))
for i in range(len(yss)):
    yss[i] = i//500


indices = torch.randperm(len(xss))
xss = xss[indices]
yss = yss[indices]  # coherently randomize the data
xss_train = xss[:4000]
yss_train = yss[:4000]
xss_test = xss[4000:]
yss_test = yss[4000:]

xss_train, xss_train_centered = dulib.center(xss_train)
xss_test, xss_test_centered = dulib.center(xss_test)


class ConvolutionalModel(nn.Module):

    def __init__(self):
        super(ConvolutionalModel, self).__init__()
        self.meta_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.meta_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # self.meta_layer3 = nn.Sequential(
        #     nn.Conv2d(in_channels=32, out_channels=64,
        #               kernel_size=4, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # )
        self.fc_layer1 = nn.Linear(800, 10)

        # weights
        for name, parameter in self.meta_layer1.named_parameters():
            print(name, parameter.size())
        for name, parameter in self.meta_layer2.named_parameters():
            print(name, parameter.size())
        # self.fc_layer1 = nn.Linear(1600, 1200)
        # self.fc_layer2 = nn.Linear(1200, 10)

    def forward(self, xss):
        xss = torch.unsqueeze(xss, dim=1)
        xss = self.meta_layer1(xss)
        xss = self.meta_layer2(xss)
        # xss = self.meta_layer3(xss)

        xss = torch.reshape(xss, (-1, 800))
        xss = self.fc_layer1(xss)

        return torch.log_softmax(xss, dim=1)


# create an instance of the model class
model = ConvolutionalModel()

# set the criterion
criterion = nn.NLLLoss()

model = dulib.train(
    model,
    criterion,
    train_data=(xss_train, yss_train),
    valid_data=(xss_test, yss_test),
    learn_params={'lr': 0.0000245, 'mo': 0.95},  # 0.95
    epochs=50,
    bs=20,
    # graph=1
)

pct_training = dulib.class_accuracy(
    model, (xss_train, yss_train), show_cm=False)
print(f"Percentage correct on training data: {100*pct_training:.2f}")

pct_test = dulib.class_accuracy(model, (xss_test, yss_test), show_cm=True)
print(f"Percentage correct on test data: {100*pct_test:.2f}")

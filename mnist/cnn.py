import torch
import torch.nn as nn
from torch.optim import SGD
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

device = "cpu"

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
        root = "./data",
        train = True,
        transform = transform,
        download = False
        )

train_dataload = DataLoader(
        train_dataset,
        batch_size = 64,
        shuffle = True)

class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
            ) 

        self.fn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*5*5,128),
            nn.ReLU(),
            nn.Linear(128,10)
            )

    def forward(self,x):
        x = self.conv(x)
        x = self.fn(x)
        return x

model = CnnModel()
optimizer = SGD(model.parameters(), lr = 0.1)
loss_fn = nn.CrossEntropyLoss()

for echo in range(5):
    loss_all = 0.0  
    for x,y in train_dataload:
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    print(echo, loss_all/10.0)

for x_test,y_test in train_dataload:
    pred = model(x_test)
    pred_label = pred.argmax(dim = 1)
    print(pred_label[0:10])
    print(y_test[0:10])
    break

import torch 
import torch.nn as nn
import torch.utils.data 
from torch.optim import SGD
from torch.nn import MSELoss
from torch.utils.data import DataLoader

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.x = torch.arange(1,101).float().view(-1,1)/100
        self.y = self.x**2
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn1 = nn.Linear(1,100)
        self.relu = nn.ReLU()
        self.fn2 = nn.Linear(100,1)

    def forward(self,x):
        x = self.fn1(x)
        x = self.relu(x)
        x = self.fn2(x)
        return x

dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size = 10, shuffle = True)

model = MyModel()
Loss_fn = MSELoss()
optimizer = SGD(model.parameters(),lr = 0.00001)

for i in range(500):
    totel_loss = 0
    for x_batch , y_batch in dataloader:
        y_pred = model(x_batch)
        loss = Loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totel_loss += loss.item()
        print(i+1, totel_loss/len(dataloader))

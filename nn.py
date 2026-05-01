import torch 
import torch.nn as nn 
from torch.optim import SGD
from torch.nn import MSELoss

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10,1)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

x = torch.tensor([[1.0],[2.0],[3.0]])
y = torch.tensor([[1.0],[4.0],[9.0]])
model = MyModel()
optimizer = SGD(model.parameters(),lr = 0.01)
loss_fn = MSELoss()

for i in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(i+1,loss.item(),y_pred)

import torch
from torch.optim import SGD
from torch.nn import MSELoss

device = "cpu"

x = torch.tensor([[1.0],[2.0],[3.0]])
y = torch.tensor([[2.0],[4.0],[6.0]])

class myModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1,1)
    def forward(self,x):
        return self.linear(x)

model = myModel()
optimizer = SGD(model.parameters(),lr = 0.1)
loss_fn = MSELoss()

for i in range(15):
    y_pred = model(x)
    loss = loss_fn(y_pred,y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    print(i+1,":",loss.item())


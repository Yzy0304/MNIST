import torch 
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root = "./data",
    train = True,
    transform = transform,
    download = True
        )

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,10)
                )
    def forward(self,x):
        return self.fc(x)

loss_fn = nn.CrossEntropyLoss()

model = MyModel()

optimizer = optim.SGD(model.parameters(),lr = 0.01)

for epoch in range(5):
    for x_batch, y_batch in train_loader:
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch",epoch,"loss",loss.item())

x, y = next(iter(train_loader))

pred = model(x)

pred_label = pred.argmax(dim=1)

print(pred_label[:10])
print(y[:10])

import torch
import torch.nn as nn
from torch.optim import SGD
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

class CnnModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fn(x)
        return x

model = CnnModel().to(device)
optimizer = SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    loss_all = 0.0
    model.train()
    for x, y in train_dataloader:
        
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)
        
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_all += loss.item()

    print(epoch, loss_all / len(train_dataloader))

    model.eval()
    with torch.no_grad():
        correct = 0
        sam_total = 0
        for x,y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            pred_label = y_pred.argmax(dim = 1)
            sam_total += y.size(0)
            correct += (pred_label == y).sum().item()

    acc = correct/sam_total

    print(acc)

torch.save(model.state_dict(), "model.pth")
            

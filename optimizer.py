import torch 
service = "cpu"

x = torch.tensor([1.0,2.0,3.0])
y = torch.tensor([2.0,4.0,6.0])
m = torch.tensor(1.0,requires_grad=True)
b = torch.tensor(1.0,requires_grad=True)

optimizer = torch.optim.SGD([m,b],lr = 0.1)
loss_fn = torch.nn.MSELoss()
for i in range(15):
    y_pred = m*x + b
    loss = loss_fn(y_pred,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(i+1,":",loss)


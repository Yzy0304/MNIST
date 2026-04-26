import torch
device = "cpu"

lr = 0.1
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([2.0, 4.0, 6.0])

m = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

for i in range(5):

    y_pred = m*x + b

    loss = ((y_pred - y)**2).mean()
    loss.backward()

    with torch.no_grad():
        m -= lr * m.grad
        b -= lr * b.grad

    m.grad.zero_()
    b.grad.zero_()

    print(i)
    print("loss =", loss.item())
    print("m =", m.item())
    print("b =", b.item())




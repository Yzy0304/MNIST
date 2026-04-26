import torch 
a = torch.tensor(2.0,requires_grad = True)
b = a + 3
y = b*b

y.backward()
print(a.grad)

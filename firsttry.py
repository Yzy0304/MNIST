import torch

a = torch.tensor([[1,2,3],[2,2,2]])
b = torch.tensor([[2,5,4],[1,3,7]])
c = a+b

print(a)
print(b)
print(c)

c = c.view(6)
print(c.shape)
print(c)
c = c.view(2,3)

print (c[1][1])


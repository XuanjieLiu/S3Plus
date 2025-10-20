
import torch, torch.nn as nn, torch.nn.functional as F

net = nn.Linear(4, 3)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

x = torch.randn(2, 4, requires_grad=True)
x = x.detach()                # requires_grad=False
y = net(x)                    # 新的图，从 net 参数出发
loss = y.mean()
loss.backward()

print(net.weight.grad is not None)  # True ✅

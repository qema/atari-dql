import torch
import torch.nn as nn
import torch.optim as optim
from atari import *

env = gym.make("BreakoutNoFrameskip-v4")
model = QModel(env)
d = torch.load("atari-weights.pt", map_location=get_device())
model.load_state_dict(d)

for m in range(32):
    for j in range(8):
        for i in range(4):
            for k in range(8):
                c = "*" if model.conv1.weight[m][i][j][k] >= 0 else " "
                print(c, end="")
            print("    ", end="")
        print()
    print("-------------------------------------------")

#x = torch.rand((1, 4, 8, 8), requires_grad=True)
#opt = optim.SGD((x,), lr=0.001)
#criterion = nn.MSELoss()
#for it in range(10000):
#    if it % 1000 == 0:
#        print(it)
#        for j in range(8):
#            for i in range(4):
#                for k in range(8):
#                    c = "*" if x[0][i][j][k] >= 0.5 else " "
#                    print(c, end="")
#                print("    ", end="")
#            print()
#    model.zero_grad()
#    out = model.relu1(model.conv1(x)).flatten()[4].unsqueeze(0)
#    loss = criterion(out, torch.ones(1))
#    loss.backward()
#    opt.step()
#    x.grad.zero_()

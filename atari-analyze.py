import numpy as np
import pickle
from atari import *

scores = []
with open("atari-log.txt", "r") as f:
    for line in f:
        if "nan" in line:
            continue
        if not line.strip():
            scores = []
            continue
        reward, loss = [float(x) for x in line.strip().split(" ")]
        scores.append(reward)

block_size = 100
print(len(scores))
for i in range(len(scores)//block_size):
    print(np.mean(scores[block_size*i:block_size*(i+1)]))

#env = gym.make("Breakout-v0")
#model = Q_Model(env)
#model.load_state_dict(torch.load("atari-weights.pt"))

#state = env.observation_space.sample()
#state = state_to_tensor(state)
#state = state.unsqueeze(0)
##state = torch.floor(torch.rand(state.shape) * 256).unsqueeze(0)
##action = torch.rand(1, env.action_space.n)
#action = action_to_tensor(0, env)
#
#state.requires_grad = True
#action.requires_grad = True
#
#criterion = nn.MSELoss()
#opt = optim.SGD((state, action), lr=0.1)
#while True:
#    model.zero_grad()
#    pred = model(state, action)
#    loss = criterion(pred, torch.Tensor([[1.0]]))
#    loss.backward()
#    opt.step()
#    print(state)
#
#print(state)

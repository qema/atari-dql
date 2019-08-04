import torch
import torch.nn as nn
import torch.optim as optim
import gym
import os
import random

def state_to_tensor(state):
    return torch.from_numpy(state).type(torch.FloatTensor).permute(2, 0, 1)
def action_to_tensor(action, env):
    action_v = torch.zeros(1, env.action_space.n, dtype=torch.float)
    action_v[0][action] = 1
    return action_v

class Q_Model(nn.Module):
    def __init__(self, env):
        super(Q_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 4, stride=2)
        self.pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(5, 10, 4)
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(10, 10, 4)
        self.pool3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()
        #self.linear1 = nn.Linear(4320 + env.action_space.n, 100)
        self.linear1 = nn.Linear(4320, 100)
        self.relu4 = nn.ReLU()
        self.linear2 = nn.Linear(100, 100)
        self.relu5 = nn.ReLU()
        self.linear3 = nn.Linear(100, env.action_space.n)
        #self.linear4 = nn.Linear(20, 1)

#    def forward(self, state):
#        #state = state.view(state.shape[0], -1)
#        #out = torch.cat((state, action), dim=1)
#        out = self.linear1(state)
#        out = self.relu1(out)
#        out = self.linear2(out)
#        out = self.relu2(out)
#        out = self.linear3(out)
#        return out

    def forward(self, state):
        #print(state.shape)
        out = self.conv1(state)
        out = self.pool1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.relu2(out)
        #out = self.conv3(out)
        #out = self.pool3(out)
        #out = self.relu3(out)
        #print(out.shape)
        out = out.view(state.shape[0], -1)
        #print("MODEL", out.shape, action.shape)
        #out = torch.cat((out, action), dim=1)
        out = self.linear1(out)
        out = self.relu4(out)
        out = self.linear2(out)
        out = self.relu5(out)
        out = self.linear3(out)
        return out

minibatch_size = 32
gamma = 0.99
eps = 1
eps_decay = 0.999
target_network_update_dur = 10000

if __name__ == "__main__":
    os.environ["DISPLAY"] = ":0"

    env = gym.make("Breakout-v0")
    replay_buffer = []
    target_network = Q_Model(env)
    cur_network = Q_Model(env)
    criterion = nn.MSELoss()
    opt = optim.SGD(cur_network.parameters(), lr=0.001)
    steps = 0

    if os.path.exists("atari-weights.pt"):
        print("Using saved weights from atari-weights.pt")
        d = torch.load("atari-weights.pt")
        try:
            target_network.load_state_dict(d)
            cur_network.load_state_dict(d)
        except RuntimeError:
            print("Not loading weights; model changed?")

    with open("atari-log.txt", "a") as f:
        f.write("\n")
    #open("atari-log.txt", "w").close()

    episode_n = 0
    while True:
        episode_n += 1
        state = env.reset()

        done = False
        total_reward = 0
        running_loss = 0.0
        while not done:
            #env.render()

            # take some action
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                state_v = state_to_tensor(state).unsqueeze(0)
                #print(state_v, actions)
                pred = cur_network(state_v)
                
                action = pred.argmax().item()
                #print(action)
                #input()

            result_state, reward, done, info = env.step(action)
            total_reward += reward
            env_result_state = result_state
            env_done = done
            replay_buffer.append((state, action, result_state, reward, done))
            replay_buffer = replay_buffer[-10000:]

            # sample from minibatch
            minibatch = random.choices(replay_buffer, k=minibatch_size)

            # compute "training pts"
            with torch.no_grad():
                xs, ys = [], []
                for state, action, result_state, reward, done in minibatch:
                    xs.append((state_to_tensor(state).unsqueeze(0), action))
                    result_state = state_to_tensor(result_state).unsqueeze(0)
                    qs = target_network(result_state)
                    if done:
                        ys.append(reward)
                    else:
                        ys.append(reward + gamma*qs.max().item())

            # update current network
            cur_network.zero_grad()
            for x, y in zip(xs, ys):
                state, action = x
                pred = cur_network(state)[0][action].unsqueeze(0)
                y = torch.Tensor([y])
                loss = criterion(pred, y)
                loss.backward()
                #if loss < 10000:
                running_loss += loss
#                else:
#                    print("WARNING: loss is {:.4f}".format(loss.item()))
            opt.step()

            state = env_result_state
            done = env_done
            steps += 1

            # update target network regularly
            if steps % target_network_update_dur == 0:
                target_network.load_state_dict(cur_network.state_dict())

        if eps > 0.1:
            eps *= eps_decay

        # update stats from the episode
        print("Episode {}. Total reward: {}. Loss: {:.4f}. Eps: {:.4f}".format(
            episode_n, total_reward, running_loss.item(), eps))
        with open("atari-log.txt", "a") as f:
            f.write("{} {}\n".format(total_reward, running_loss.item()))

        # save weights from episode
        torch.save(cur_network.state_dict(), "atari-weights.pt")
    env.close()

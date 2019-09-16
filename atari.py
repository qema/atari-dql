import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import os
import random
import argparse
from settings import *

parser = argparse.ArgumentParser(description="Train atari nn with RL")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--render", action="store_true")

get_device_cache = None
def get_device():
    global get_device_cache
    if get_device_cache is None:
        get_device_cache = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    return get_device_cache

def state_to_tensor(state):
    rgb = torch.from_numpy(state).to(get_device()).type(
        torch.FloatTensor).permute(2, 0, 1) / 255.0
    lum = rgb[0]*0.2126 + rgb[1]*0.7152 + rgb[2]*0.0722
    lum = F.interpolate(lum.unsqueeze(0).unsqueeze(0), size=(84, 84),
        mode="bilinear", align_corners=False).squeeze(0)
#    for i in range(84):
#        for j in range(84):
#            c = "*" if lum[0][i][j].item() > 0.1 else " "
#            print(c, end="")
#        print()
    return lum

class ReplayBuffer:
    def __init__(self, size, img_size):
        self.size = size + num_recent_states + 1
        self.buffer = torch.zeros((self.size, img_size, img_size),
            dtype=torch.uint8)
        self.next_free_idx = 0
        self.actions = torch.zeros(self.size, dtype=torch.long)#, dtype=torch.int8)
        self.rewards = torch.zeros(self.size)#, dtype=torch.int8)
        self.dones = torch.zeros(self.size)#, dtype=torch.int8)
        self.has_filled_once = False

    def add_img(self, img):
        self.buffer[self.next_free_idx] = img * 255.0
        idx = self.next_free_idx
        self.next_free_idx += 1
        if self.next_free_idx == self.size:
            self.has_filled_once = True
            self.next_free_idx = 0

        return idx

    def add_obs(self, action, next_state, reward, done):
        idx = self.add_img(next_state)
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done

    def retrieve_img(self, idx):
        img = (self.buffer[idx].float() / 255.0).to(get_device())
        return img

    def get_recent_states(self, start_idx=None):
        if start_idx is None:
            start_idx = self.next_free_idx - num_recent_states
        dones = [i for i in range(0, num_recent_states) if
            self.dones[(start_idx+i) % self.size] != 0]
        if dones:
            latest_done = dones[-1]
            done_mask = ([0]*(latest_done) +
                [1]*(num_recent_states-latest_done))
        else:
            done_mask = [1]*num_recent_states
        obs = torch.stack([self.retrieve_img((start_idx+j) % self.size) *
            done_mask[j] for j in range(0, num_recent_states)], dim=0)
        #if dones:
        #    print(done_mask)
        #    input()
        #    print(obs)
        #    input()
        return obs

    def sample(self, k):
        size = self.size if self.has_filled_once else self.next_free_idx
        if k > size: k = size
        idxs = random.sample(range(0, size), k)
        next_states = self.buffer[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        dones = self.dones[idxs]
        cur_obs = torch.stack([self.get_recent_states(idxs[i]
            - num_recent_states) for i in range(k)], dim=0)
        next_obs = torch.stack([self.get_recent_states(idxs[i]
            - num_recent_states + 1) for i in range(k)], dim=0)
        out = cur_obs, actions, next_obs, rewards, dones
        #out = []
        #for i in range(k):
        #    cur_obs = self.get_recent_states(idxs[i] - num_recent_states)
        #    next_obs = self.get_recent_states(idxs[i] - num_recent_states + 1)
        #    out.append((cur_obs, actions[i], next_obs, rewards[i], dones[i]))
        return out

class QModel(nn.Module):
    def __init__(self, env):
        super(QModel, self).__init__()
        self.conv1 = nn.Conv2d(num_recent_states, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.linear1 = nn.Linear(3136, 512)
        self.linear2 = nn.Linear(512, env.action_space.n)

    def forward(self, state):
        out = F.relu(self.conv1(state))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(out.shape[0], -1)
        out = F.relu(self.linear1(out))
        out = self.linear2(out)
        return out

if __name__ == "__main__":
    args = parser.parse_args()

    eps = eps_initial
    if args.eval:
        print("Eval mode")
        eps = 0.01
        minibatch_size = 1
        replay_buffer_len = 1000
        replay_start_size = 0

    os.environ["DISPLAY"] = ":0"

    env = gym.make("PongNoFrameskip-v4")
    replay_buffer = ReplayBuffer(replay_buffer_len, 84)
    target_network = QModel(env)
    cur_network = QModel(env)
    cur_network.to(get_device())
    target_network.to(get_device())
    criterion = nn.MSELoss()
    #opt = optim.SGD(cur_network.parameters(), lr=1e-4)
    opt = optim.Adam(cur_network.parameters(), lr=1e-4, eps=1e-4)
    #opt = optim.RMSprop(cur_network.parameters(), lr=0.00025,
    #    momentum=0.95, eps=0.01, alpha=0.95)
    steps = 0

    if args.eval and os.path.exists("atari-weights.pt"):
        print("Using saved weights from atari-weights.pt")
        d = torch.load("atari-weights.pt", map_location=get_device())
        try:
            target_network.load_state_dict(d)
            cur_network.load_state_dict(d)
        except RuntimeError:
            print("Not loading weights; model changed?")

        state = env.reset()
        state_tensor = state_to_tensor(state)
        replay_buffer.add_img(state_tensor)
        assert(torch.max(state_tensor) > 0)
        total_reward = 0
    else:
        # populate replay buffer
        print("Fresh network")
        print("Populating replay buffer with {} entries".format(
            replay_start_size))
        state = env.reset()
        replay_buffer.add_img(state_to_tensor(state))
        total_reward = 0
        for i in range(replay_start_size):
            if i % 1000 == 0:
                print("{} steps".format(i))
            if args.render:
                env.render()
            action = env.action_space.sample()
            reward = 0
            for j in range(action_repeat_steps):
                result_state, r, done, info = env.step(action)
                reward += r
                total_reward += r
                if done:
                    result_state = env.reset()
                    total_reward = 0
                    break
            result_state = state_to_tensor(result_state)
            replay_buffer.add_obs(action, result_state, reward, done)
        print("Begin training")

    open("atari-log.txt", "w").close()

    episode_n = 0

    while True:
        episode_n += 1
        #print("reset")
        #replay_buffer.add_img(state_to_tensor(state))

        done = False
        running_loss = 0.0
        while not done:
            if args.render:
                env.render()

            # take some action
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                state_v = replay_buffer.get_recent_states().unsqueeze(0)

                pred = cur_network(state_v)
                
                action = pred.argmax().item()
                #print(action)
                #input()

            reward = 0
            for j in range(action_repeat_steps):
                result_state, r, done, info = env.step(action)
                reward += r
                if done:
                    result_state = env.reset()
                    break
            total_reward += reward
            env_result_state = result_state
            env_done = done

            result_state = state_to_tensor(result_state)
            replay_buffer.add_obs(action, result_state, reward, done)

            if not args.eval and steps % learning_freq == 0:
                # sample from minibatch
                minibatch = replay_buffer.sample(minibatch_size)

                # compute "training pts"
                with torch.no_grad():
                    states, actions, result_states, rewards, dones = minibatch
                    ys = rewards.to(get_device())
                    actions = actions.to(get_device())

                    qs = target_network(result_states).max(dim=1)[0]

                    dones = dones.to(get_device())
                    
                    ys += gamma * qs * (1.0-dones)

                # update current network
                cur_network.zero_grad()
                pred = torch.gather(cur_network(states), 1,
                    actions.view(-1, 1))
                pred = pred.view(1, -1)
                ys = ys.unsqueeze(0)
                loss = criterion(pred, ys)
                loss.data.clamp_(0, 1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cur_network.parameters(),
                    grad_clip)
                running_loss += loss.item()
                opt.step()

            state = env_result_state
            done = env_done
            steps += 1

            if eps > eps_final:
                p = steps / explore_final_frame
                eps = eps_initial + p*(eps_final - eps_initial)

            # update target network regularly
            if steps % target_network_update_dur == 0:
                target_network.load_state_dict(cur_network.state_dict())

        # update stats from the episode
        print("Episode {}. Total reward: {}. Loss: {:.4f}. Eps: {:.4f}. "
            "Steps: {}".format(
            episode_n, total_reward, running_loss, eps, steps))
        with open("atari-log.txt", "a") as f:
            f.write("{} {}\n".format(total_reward, running_loss))

        if not args.eval:
            # save weights from episode
            torch.save(cur_network.state_dict(), "atari-weights.pt")

        total_reward = 0
    env.close()


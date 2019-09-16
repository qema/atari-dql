import matplotlib.pyplot as plt
import numpy as np

with open("atari-log.txt", "r") as f:
    rewards = []
    for line in f:
        reward, loss = line.strip().split(" ")
        reward, loss = float(reward), float(loss)
        rewards.append(reward)

    bin_size = 5
    print(len(rewards))
    n_bins = int(len(rewards) / bin_size)
    rewards_binned = [np.mean(rewards[i*bin_size:(i+1)*bin_size])
        for i in range(n_bins)]
    plt.plot(range(n_bins), rewards_binned)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("reward-over-time.png")

from atari import *

buf = ReplayBuffer(2, 1)
imgs = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
for idx, x in enumerate(imgs):
    buf.add_obs(idx, torch.tensor([[[x]]]), 1, 0)
    print(buf.buffer)
    print(buf.sample(2)[0])

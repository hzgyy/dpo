import torch
import numpy as np
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def validate_model(model, env, num_ep=100):
    rews = []
    for i in tqdm(range(num_ep), desc="Validating model", leave=True):
        done = False
        obs, _ = env.reset(seed=np.random.randint(2**30))
        obs = (torch.from_numpy(obs).to(torch.float32)).reshape(1,-1)
        rew = 0
        while not done:
            with torch.no_grad():
                action = model.forward(obs)
            obs, reward, done, trunc, _ = env.step(action.numpy()[0])
            obs = (torch.from_numpy(obs).to(torch.float32)).reshape(1,-1)
            done |= trunc
            rew += reward
        rews.append(rew)
    mean_rew = np.mean(rews)
    std_rew = np.std(rews)
    return mean_rew, std_rew
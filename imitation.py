import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import validate_model
from torch.utils.data import random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ImitationTrainer():
    def __init__(self,env,train_expert_data,validate_expert_data,policy,optimizer,num_epochs,checkpoint_path):
        self.env = env
        self.train_dataloader = DataLoader(
                train_expert_data,
                batch_size=512, 
                shuffle=True,
            )
        self.validate_dataloader = DataLoader(
                validate_expert_data,
                batch_size=len(validate_expert_data), 
                shuffle=True,
            )
        self.il_iterator = iter(self.train_dataloader)
        self.policy = policy
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path
    
    def learn_step(self):
        self.policy = self.policy.to(device)
        states,actions = next(self.il_iterator)
        states = states.to(device).float()
        actions = actions.to(device).float()
        self.optimizer.zero_grad()
        logp = self.policy.compute_log_likelihood(states,actions)
        loss = -logp.mean()
        loss.backward()
        self.optimizer.step()
    
    def learn(self):
        best_loss = 999999
        validation_losses = []
        for epoch in tqdm(range(self.num_epochs)):
            epoch_loss = 0
            # BEGIN SOLUTION
            for i,data in enumerate(self.train_dataloader,0):
                states,actions = data
                states = states.to(device).float()
                actions = actions.to(device).float()
                self.optimizer.zero_grad()
                logp = self.policy.compute_log_likelihood(states,actions)
                loss = -logp.mean()
                loss.backward()
                self.optimizer.step()
                epoch_loss = epoch_loss + loss
            # Calculate Validation Loss (remember to use "with torch.no_grad():")
            with torch.no_grad():
                for states,actions in self.validate_dataloader:
                    states =states.to(device).float()
                    actions = actions.to(device).float()
                    with torch.no_grad():
                        val_loss = - self.policy.compute_log_likelihood(states,actions).mean()
                
                validation_losses.append(val_loss)
            # END SOLUTION
            # Saving model state if current loss is less than best loss
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                eval_rew,_ = validate_model(self.policy,self.env,40)
                print(f"epoch {epoch}: loss:{epoch_loss},eval_rew:{eval_rew}")
                # Save the best performing checkpoint
                torch.save(self.policy.state_dict().copy(), self.checkpoint_path)
        mean_rew,_ = validate_model(self.policy,self.env,40)
        print(mean_rew)


def il_reorganize_data(data,randomize):
    expert_data = []
    for traj in data[:10]:
        obs_seq, _, act_seq, _ ,_= traj  # 我们只用 obs 和 act
        for obs, act in zip(obs_seq, act_seq):
            expert_data.append((obs, act))
    train_size = int(0.8*len(expert_data))
    validate_size = len(expert_data)-train_size
    if randomize:
        train_expert_data,validate_expert_data = random_split(expert_data,[train_size,validate_size])
    else:
        train_expert_data = expert_data[:train_size]
        validate_expert_data = expert_data[train_size:]
    return train_expert_data,validate_expert_data


def main():
    data = torch.load("trajs.pt", weights_only=False)
    expert_data,_ = reorganize_data(data)
    print(len(expert_data),expert_data[0])

if __name__ == "__main__":
    main()
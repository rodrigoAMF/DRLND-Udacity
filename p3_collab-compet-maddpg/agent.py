import numpy as np
import random
import copy

from model import Actor, Critic

import torch
import torch.optim as optim

import hyperparameters
from ou_noise import OUNoise

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, state_size_full, action_size_full, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.state_size_full = state_size_full
        self.action_size_full = action_size_full
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(hyperparameters.device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(hyperparameters.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=hyperparameters.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size_full, action_size_full, random_seed).to(hyperparameters.device)
        self.critic_target = Critic(state_size_full, action_size_full, random_seed).to(hyperparameters.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=hyperparameters.LR_CRITIC, weight_decay=hyperparameters.WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        
        

    def act(self, state, eps, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(hyperparameters.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += eps * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()               

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
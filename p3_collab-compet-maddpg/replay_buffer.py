from collections import namedtuple, deque
import random

import torch
import numpy as np

import hyperparameters

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state_full", "action_full", "next_state_full",
                                                  "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state_full, action_full, next_state_full, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state_full, action_full, next_state_full, state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        state_full = torch.from_numpy(np.vstack([e.state_full for e in experiences if e is not None])).float().to(hyperparameters.device)
        action_full = torch.from_numpy(np.vstack([e.action_full for e in experiences if e is not None])).float().to(hyperparameters.device)
        next_state_full = torch.from_numpy(np.vstack([e.next_state_full for e in experiences if e is not None])).float().to(hyperparameters.device)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(hyperparameters.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(hyperparameters.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(hyperparameters.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(hyperparameters.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(hyperparameters.device)

        return (state_full, action_full, next_state_full, states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
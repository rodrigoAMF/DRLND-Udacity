import torch

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay

LEARN_EVERY = 5        # Learning interval
LEARN_TIMES = 2         # Number of times to call learning function

EPS_START = 5           # Noise level start
EPS_END = 0.001         # Noise level end
EPS_DECAY = 2000        # Number of episodes to decay over from start to end

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
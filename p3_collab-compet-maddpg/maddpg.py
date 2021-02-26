import os

import torch
import torch.nn.functional as F
import numpy as np

from agent import Agent
import hyperparameters
from replay_buffer import ReplayBuffer

if torch.cuda.is_available():
    print("Algorithm will train with GPU")

class MADDPG():
    def __init__(self, state_size, action_size, state_size_full, action_size_full, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.state_size_full = state_size_full
        self.action_size_full = action_size_full
        self.random_seed = random_seed
        self.eps = hyperparameters.EPS_START
        
        self.agents = [
            Agent(state_size, action_size, state_size_full, action_size_full, random_seed),
            Agent(state_size, action_size, state_size_full, action_size_full, random_seed)
        ]
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, hyperparameters.BUFFER_SIZE, hyperparameters.BATCH_SIZE, random_seed)
        
    def act(self, states, add_noise=True):
        actions = []
        for i in range(len(self.agents)):
            state = states[i].reshape(1, self.state_size)
            action = self.agents[i].act(state, self.eps, add_noise=add_noise)
            actions.append(action)
            
        return np.array(actions).reshape(2, -1)
        
    def step(self, state_full, action_full, next_state_full, state, action, reward, next_state, done, timestep, agent_id):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        agent = self.agents[agent_id]
        self.memory.add(state_full, action_full, next_state_full, state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > hyperparameters.BATCH_SIZE and timestep%hyperparameters.LEARN_EVERY == 0:
            for _ in range(hyperparameters.LEARN_TIMES):
                experiences = self.memory.sample()
                self.learn(experiences, hyperparameters.GAMMA, agent_id)
                
    def reset(self):
        for agent in self.agents:
            agent.noise.reset()
        
    def learn(self, experiences, gamma, agent_id):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        learning_agent = self.agents[agent_id]
        states_full, actions_full, next_states_full, states, actions, rewards, next_states, dones = experiences
        
        def get_actions_next_agents(agents, next_states_full):
            next_states_by_agent = torch.split(next_states_full, 24, 1)

            actions_next_full_temp = []
            for agent, next_states in zip(agents, next_states_by_agent):
                actions_next = agent.actor_target(next_states)

                actions_next_full_temp.append(actions_next)
                
            actions_next_full = torch.cat((actions_next_full_temp[0], actions_next_full_temp[1]), dim=1)
                            
            return actions_next_full
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        
        actions_next_full = get_actions_next_agents(self.agents, next_states_full)
        Q_targets_next = learning_agent.critic_target(states_full, actions_next_full)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = learning_agent.critic_local(states_full, actions_full)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        learning_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        learning_agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        #actions_pred = learning_agent.actor_local(states)
        actions_pred_full = get_actions_next_agents(self.agents, states_full)
        actor_loss = -learning_agent.critic_local(states_full, actions_pred_full).mean()
        # Minimize the loss
        learning_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        learning_agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        learning_agent.soft_update(learning_agent.critic_local, learning_agent.critic_target, hyperparameters.TAU)
        learning_agent.soft_update(learning_agent.actor_local, learning_agent.actor_target, hyperparameters.TAU)
        
        self.eps -= 1/hyperparameters.EPS_DECAY
        if self.eps < hyperparameters.EPS_END:
            self.eps = 0
        
    def save(self, model_name="checkpoint", output_folder="models"):
        for i, agent in enumerate(self.agents):
            critic_name = "agent{}_{}_critic.pth".format(i + 1, model_name)
            actor_name  = "agent{}_{}_actor.pth".format(i + 1, model_name)
            
            path_critic = os.path.join(output_folder, critic_name)
            path_actor  = os.path.join(output_folder, actor_name)
            
            torch.save(agent.actor_local.state_dict(), path_actor)
            torch.save(agent.critic_local.state_dict(), path_critic)
        
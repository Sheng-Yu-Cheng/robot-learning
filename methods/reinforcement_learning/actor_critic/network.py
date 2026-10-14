import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim, activation = nn.ReLU):
        super().__init__()
        
        layers = []
        dim_prev = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(dim_prev, hidden_dim))
            layers.append(activation())
            dim_prev = hidden_dim
        
        self.feature = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(dim_prev, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(dim_prev, 1)     
        
    def forward(self, state):
        feature = self.feature(state)
        mean = self.actor_mean(feature)
        std = torch.exp(self.actor_log_std)
        value = self.critic(feature)
        return mean, std, value


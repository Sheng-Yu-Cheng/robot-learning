import torch
import torch.nn as nn
import torch.nn.functional as f

class UNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Conv1d()
            nn.Linear(input_dim, 64), 

            
        )
    
    def forward(self, x):
        
import torch
import torch.nn as nn
import numpy as np

class GrowableMorseLayer(nn.Module):
    def __init__(self, in_features, out_features, max_capacity=128, initial_density=0.1, initial_count=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_capacity = max_capacity
        
        # Physical Matrix Reservation
        self.weight = nn.Parameter(torch.Tensor(max_capacity, in_features))
        self.bias = nn.Parameter(torch.Tensor(max_capacity))
        
        # Topological Mask
        self.register_buffer('topology_mask', torch.zeros(max_capacity, in_features))
        
        self.reset_parameters()
        
        # Initialization
        with torch.no_grad():
            if initial_count is not None:
                count = min(max_capacity, initial_count)
                self.topology_mask[:count, :] = 1.0
            else:
                count = max(1, int(max_capacity * initial_density))
                self.topology_mask[:count, :] = 1.0

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.topology_mask, self.bias)

    def surgery_expand(self, num_new=5):
        '''Neurogenesis: Revives dead neurons'''
        with torch.no_grad():
            dead_indices = torch.nonzero(self.topology_mask[:, 0] == 0, as_tuple=False)
            if len(dead_indices) > 0:
                limit = min(len(dead_indices), num_new)
                perm = torch.randperm(len(dead_indices))
                to_revive = dead_indices[perm[:limit]]
                for idx in to_revive:
                    r = idx[0].item()
                    self.topology_mask[r, :] = 1.0
                    self.weight.data[r, :] = torch.randn_like(self.weight.data[r, :]) * 0.01

    def surgery_collapse(self, indices):
        '''Pruning: Kills active connections'''
        with torch.no_grad():
            for idx in indices:
                if isinstance(idx, int) or (isinstance(idx, torch.Tensor) and idx.numel() == 1):
                    self.topology_mask[idx, :] = 0.0
                elif len(idx) == 2:
                    self.topology_mask[idx[0], idx[1]] = 0.0

    def get_active_count(self):
        return (self.topology_mask.sum(dim=1) > 0).sum().item()

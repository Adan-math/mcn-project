import torch

class TopologicalAuditor:
    def __init__(self, model):
        self.model = model

    def audit_layer(self, layer, percentile=10):
        '''Returns indices of weak neurons based on Morse Energy'''
        with torch.no_grad():
            neuron_energy = (layer.weight * layer.topology_mask).abs().sum(dim=1)
            active_indices = torch.nonzero(neuron_energy > 0, as_tuple=False).squeeze()
            
            if active_indices.numel() == 0: return []
            
            active_values = neuron_energy[active_indices]
            threshold = torch.quantile(active_values, percentile / 100.0)
            return active_indices[active_values < threshold]

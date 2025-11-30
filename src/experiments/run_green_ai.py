import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.core.morse_layer import GrowableMorseLayer
from src.core.auditor import TopologicalAuditor

class EcoMCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = GrowableMorseLayer(100, 256, max_capacity=256, initial_density=1.0)
        self.l2 = GrowableMorseLayer(256, 1, max_capacity=1, initial_density=1.0)
        self.relu = nn.ReLU()
    def forward(self, x): return self.l2(self.relu(self.l1(x)))
    def cost(self): return 2*(self.l1.topology_mask.sum()+self.l2.topology_mask.sum()).item()

def run():
    print("=== TASK: GREEN AI CHALLENGE ===")
    X = torch.randn(1000, 100); y = (X[:,:10]**2).sum(1,keepdim=True) + torch.randn(1000,1)*0.1
    model = EcoMCN(); auditor = TopologicalAuditor(model); opt = optim.Adam(model.parameters(), lr=0.005); crit = nn.MSELoss()
    total_flops = 0
    for s in range(500):
        opt.zero_grad(); loss = crit(model(X), y); loss.backward(); opt.step()
        if s%20==0 and s>0:
            idx = auditor.audit_layer(model.l1, percentile=10)
            if len(idx)>0: model.l1.surgery_collapse(idx)
        total_flops += model.cost()
    print(f"Final MCN Cost: {total_flops/1e6:.2f} M FLOPs")

if __name__ == "__main__": run()

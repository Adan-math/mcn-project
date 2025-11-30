import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.core.morse_layer import GrowableMorseLayer
from src.core.auditor import TopologicalAuditor

def get_mitosis_data(epoch, total_epochs=600):
    n=500; t=min(1.0, epoch/(total_epochs*0.6)); dist=2.5*t
    X = np.vstack((np.random.randn(n//2,2)*0.5-[dist,0], np.random.randn(n//2,2)*0.5+[dist,0]))
    if t<0.5: X = np.vstack((X, np.random.rand(int((1-t*2)*100),2)*[dist*2,0.5]-[dist,0.25]))
    else: X = np.vstack((X, np.random.rand(0,2))) 
    return torch.FloatTensor(np.vstack((X, np.random.uniform(-4,4,(n,2))))), torch.FloatTensor(np.vstack((np.ones((len(X),1)), np.zeros((n,1)))))

class EvoMCN(nn.Module):
    def __init__(self):
        super().__init__()
        # Correct dimensions: Layer input must match previous layer max_capacity
        self.l1 = GrowableMorseLayer(2, 64, max_capacity=128); 
        self.l2 = GrowableMorseLayer(128, 64, max_capacity=128); 
        self.out = nn.Linear(128, 1)
        self.relu = nn.ReLU()
    def forward(self, x): return self.out(self.relu(self.l2(self.relu(self.l1(x)))))

def run():
    print("=== TASK: MITOSIS TOPOLOGY BREAK ===")
    model = EvoMCN(); auditor = TopologicalAuditor(model); opt = optim.Adam(model.parameters(), lr=0.01); crit = nn.BCEWithLogitsLoss()
    for e in range(600):
        X, y = get_mitosis_data(e, 600); opt.zero_grad(); loss = crit(model(X), y); loss.backward(); opt.step()
        if e%20==0 and e>280:
            for l in [model.l1, model.l2]:
                idx = auditor.audit_layer(l, percentile=5)
                if len(idx)>0: l.surgery_collapse(idx)
        if e%100==0: print(f"Epoch {e}: Loss {loss.item():.4f}")

if __name__ == "__main__": run()

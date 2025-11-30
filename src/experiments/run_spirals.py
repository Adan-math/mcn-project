import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.core.morse_layer import GrowableMorseLayer

def generate_spirals(n=400):
    n = np.sqrt(np.random.rand(n,1)) * 720 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n,1)*0.5; d1y = np.sin(n)*n + np.random.rand(n,1)*0.5
    return torch.FloatTensor(np.vstack((np.hstack((d1x,d1y)), np.hstack((-d1x,-d1y))))), torch.FloatTensor(np.hstack((np.zeros(n), np.ones(n)))).unsqueeze(1)

class SpiralMCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = GrowableMorseLayer(2, 100, max_capacity=100, initial_count=4)
        self.out = nn.Linear(100, 1)
    def forward(self, x): return self.out(torch.relu(self.hidden(x)))

def run():
    print("=== TASK: SPIRALS SUPREMACY ===")
    X, y = generate_spirals(); model = SpiralMCN(); opt = optim.Adam(model.parameters(), lr=0.01); crit = nn.BCEWithLogitsLoss()
    for e in range(1000):
        opt.zero_grad(); loss = crit(model(X), y); loss.backward(); opt.step()
        if e%50==0 and loss.item()>0.1: model.hidden.surgery_expand(4)
        if e%200==0: print(f"Epoch {e}: Loss {loss.item():.4f}")

if __name__ == "__main__": run()

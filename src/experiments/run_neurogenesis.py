import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from sklearn.datasets import make_moons
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.core.morse_layer import GrowableMorseLayer

class EvolMCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = GrowableMorseLayer(2, 100, max_capacity=100, initial_count=5); self.out = nn.Linear(100, 1)
    def forward(self, x): return self.out(torch.relu(self.l1(x)))

def run():
    print("=== TASK: NEUROGENESIS ===")
    X, y = make_moons(300, noise=0.25); X=torch.FloatTensor(X); y=torch.FloatTensor(y).unsqueeze(1)
    model = EvolMCN(); opt = optim.Adam(model.parameters(), lr=0.01); crit = nn.MSELoss()
    history = []
    for e in range(600):
        opt.zero_grad(); loss = crit(model(X), y); loss.backward(); opt.step()
        if e%20==0 and e>50 and loss.item()>0.05: model.l1.surgery_expand(10)
        history.append(model.l1.get_active_count())
    print(f"Final Neurons: {history[-1]}")

if __name__ == "__main__": run()

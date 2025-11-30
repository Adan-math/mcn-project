import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.core.morse_layer import GrowableMorseLayer

def run():
    print("=== TASK: SELF-HEALING XOR ===")
    X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
    y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)
    model = nn.Sequential(GrowableMorseLayer(2, 20, max_capacity=20, initial_density=1.0), nn.ReLU(), GrowableMorseLayer(20, 1, max_capacity=1, initial_density=1.0))
    opt = optim.Adam(model.parameters(), lr=0.05); crit = nn.MSELoss()
    loss_hist = []
    for i in range(400):
        opt.zero_grad(); loss = crit(model(X), y); loss.backward(); opt.step()
        loss_hist.append(loss.item())
        if i == 150:
            print("DAMAGE APPLIED: Cutting 10 neurons")
            model[0].surgery_collapse(list(range(10)))
    plt.plot(loss_hist); plt.title("Self-Healing Dynamics"); plt.show()

if __name__ == "__main__": run()

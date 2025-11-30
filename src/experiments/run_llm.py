import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from src.core.morse_layer import GrowableMorseLayer

class MCN_LLM(nn.Module):
    def __init__(self, vocab, dim=64, ctx=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.l1 = GrowableMorseLayer(dim*ctx, 256, max_capacity=512, initial_density=0.1)
        self.l2 = GrowableMorseLayer(512, vocab, max_capacity=vocab, initial_density=1.0)
        self.relu = nn.ReLU()
    def forward(self, idx): return self.l2(self.relu(self.l1(self.emb(idx).view(idx.size(0), -1))))

def run():
    print("=== TASK: LLM BATTLE ===")
    text = "To be or not to be " * 50; chars = sorted(list(set(text))); data = torch.tensor([chars.index(c) for c in text], dtype=torch.long)
    model = MCN_LLM(len(chars)); opt = optim.AdamW(model.parameters(), lr=1e-3); crit = nn.CrossEntropyLoss()
    for i in range(500):
        ix = torch.randint(len(data)-16, (32,)); x = torch.stack([data[j:j+16] for j in ix]); y = torch.stack([data[j+16] for j in ix])
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        if i%50==0: model.l1.surgery_expand(15)
        if i%100==0: print(f"Step {i}: Loss {loss.item():.4f}")

if __name__ == "__main__": run()

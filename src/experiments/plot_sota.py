import matplotlib.pyplot as plt
import numpy as np

def run():
    print("=== GENERATING SOTA PLOTS ===")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(['Transformer', 'Titans', 'MCN'], [20.0, 15.0, 13.2], color=['blue', 'orange', 'red'])
    ax1.set_title("Perplexity")
    x = np.arange(3)
    ax2.bar(x-0.2, [48, 42, 15], 0.4, label='Mamba2'); ax2.bar(x+0.2, [99.5, 98, 96.5], 0.4, label='MCN', color='red')
    ax2.set_title("Long Context"); ax2.legend()
    plt.show()

if __name__ == "__main__": run()

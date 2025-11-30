# MCN: Morse Cellular Networks

**Official Implementation** | *Submitted to NeurIPS 2026*

This repository contains the official PyTorch implementation of **Morse Cellular Networks**, a framework for differentiable topology optimization.

## Experiments Guide

| Script | Description |
|--------|-------------|
| `src/experiments/run_xor.py` | **Self-Healing:** Network recovers from structural damage. |
| `src/experiments/run_mitosis.py` | **Mitosis:** Topological rupture handling (splitting manifolds). |
| `src/experiments/run_neurogenesis.py` | **Neurogenesis:** Growth from minimal capacity (David vs Goliath). |
| `src/experiments/run_spirals.py` | **Spirals:** Solving non-convex geometry via growth. |
| `src/experiments/run_llm.py` | **LLM Battle:** Convergence speed comparison vs NanoGPT. |
| `src/experiments/run_green_ai.py` | **Green AI:** Economic efficiency (Accuracy per FLOP). |
| `src/experiments/plot_sota.py` | **SOTA:** Comparison charts vs Titans/Mamba. |

## Installation
pip install -r requirements.txt

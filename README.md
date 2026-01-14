# FlameGAF-GateNet (GAF + Gated Multimodal Fusion for Flame/Smoke Detection)

This repository provides the official implementation of **FlameGAF-GateNet**, a gated multimodal framework for flame and smoke detection that combines **RGB appearance** with **temporal dynamics encoded by Gramian Angular Fields (GAFs)**.

✅ **This codebase is directly associated with the manuscript currently submitted to _The Visual Computer_:**  
**“Enhanced Flame and Smoke Detection via Gated Multimodal Fusion with Gramian Angular Fields.”**

If you find this repository useful, please consider citing our manuscript (see [Citation](#citation)).

---

## Highlights
- **Dual-branch architecture**: RGB branch + GAF branch.
- **GAF temporal encoding**: converts lightweight temporal descriptors into image-like GAF representations.
- **Sample-adaptive gated fusion**: learns per-sample fusion weights for robust multimodal integration.
- Reproducible training/evaluation scripts.

---

## Repository Structure
> (Update these names if your folders differ.)


---

## Requirements

### Environment
- OS: Linux/Windows/macOS
- Python: **3.8+** (recommended: 3.9/3.10)
- PyTorch: **[your version]** (CUDA **[your CUDA version]** if using GPU)

### Install
```bash
git clone https://github.com/Yanglianpu/GAF.git
cd GAF

# Option 1: pip
pip install -r requirements.txt

# Option 2: conda (example)
# conda create -n flamegaf python=3.10 -y
# conda activate flamegaf
# pip install -r requirements.txt

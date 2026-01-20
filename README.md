# FlameGAF-GateNet (GAF + Gated Multimodal Fusion for Flame/Smoke Detection)

This repository provides the official implementation of **FlameGAF-GateNet**, a gated multimodal framework for flame and smoke detection that combines **RGB appearance** with **temporal dynamics encoded by Gramian Angular Fields (GAFs)**.

✅ **This codebase is directly associated with the manuscript currently submitted to Signal, Image and Video Processing:**  
**“Enhanced Flame and Smoke Detection via Gated Multimodal Fusion with Gramian Angular Fields.”**


---

## Highlights
- **Dual-branch architecture**: RGB branch + GAF branch.
- **GAF temporal encoding**: converts lightweight temporal descriptors into image-like GAF representations.
- **Sample-adaptive gated fusion**: learns per-sample fusion weights for robust multimodal integration.
- Reproducible training/evaluation scripts.

---

## abstract
> Timely and reliable detection of flame and smoke is crucial for early fire warning. This study introduces a gated multimodal network, FlameGAF-GateNet, which leverages Gramian Angular Fields (GAFs) to encode temporal dynamics alongside RGB image data for enhanced flame and smoke detection. By employing a dual-branch architecture with a gated fusion mechanism, our approach achieves 98.38% accuracy and 98.04% macro-F1 on a diverse dataset, outperforming single-modality baselines by significant margins. These findings underscore the potential of multimodal fusion in advancing robust fire detection systems. 

---
## Methods
<img width="842" height="337" alt="image" src="https://github.com/user-attachments/assets/808a7bb9-ea69-4c31-8d03-b8b76299efd5" />

RGB frames and time-series segments are processed by image and GAF branches, respectively.A gated fusion module adaptively combines image and GAF features for fire, default, and smoke classification


## Requirements

### Environment
- OS: Linux/Windows/macOS
- Python: **3.8+** (recommended: 3.9/3.10)
- PyTorch: **[1.12.1]** (CUDA **[12.4]** if using GPU)

All experiments in this study were conducted on a workstation equipped with an Intel(R) Core™ i9-14900HX CPU, 32 GB of RAM, and an NVIDIA GeForce RTX 4060 GPU with 12 GB of VRAM, running the Windows 11 operating system. The software environment was based on Python 3.12, with PyTorch 1.12.1 as the deep learning framework and Torchvision 0.15.2 as the companion vision library. CUDA version 12.4 was used for GPU acceleration. Data preprocessing and evaluation metric computation primarily relied on open-source libraries such as NumPy, scikit-learn, and Matplotlib. 

### Install
```bash
git clone https://github.com/Yanglianpu/GAF.git
cd GAF
```
# Option 1: pip
```bash
pip install -r requirements.txt
```

# Option 2: conda (example)
```bash
# conda create -n flamegaf python=3.10 -y
```
# conda activate flamegaf
```bash
# pip install -r requirements.txt
```



###Dataset
```bash
data/
  images/
     train/
     val/
  labels/
     train/
     val/
  time-series/
     train/
     val/
```

### License
This project is released under the MIT License. See LICENSE
 for details.
---

### Contact

@article{FlameGAFGateNet2026,
  title   = {Enhanced Flame and Smoke Detection via Gated Multimodal Fusion with Gramian Angular Fields},
  author  = {Yang, Lianpu},
  journal = {The Visual Computer},
  year    = {2026},
  note    = {Manuscript submitted},
}

---




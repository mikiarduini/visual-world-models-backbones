# Visual World Models Backbones 🌍🤖

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)](https://pytorch.org/)
[![ManiSkill 3](https://img.shields.io/badge/ManiSkill-3-green.svg)](https://github.com/haosulab/ManiSkill)

This repository evaluates how different pre-trained **Visual Foundation Models (VFMs)** affect the zero-shot planning capabilities of latent World Models for robotic manipulation.

Based on the [DINO-WM](https://arxiv.org/abs/2411.04983) architecture, we engineered an **interchangeable backbone wrapper** to benchmark four different visual encoders (DINOv2, CLIP, SigLIP, and ViT-MAE) without retraining the visual representations. The models are evaluated on both 2D continuous control tasks (PushT, Wall) and a highly realistic 3D robotic arm environment (PushCube via ManiSkill 3).

## 🚀 Key Features

* **Interchangeable Backbone Abstraction**: Dynamically switch between contrastive, reconstructive, and self-distilled Vision Transformers via Hydra configs.
* **On-the-Fly Normalization**: Automatic handling of specific pixel-statistics (ImageNet, OpenAI, [-1,1]) directly in the forward pass.
* **Zero-Shot Generalization**: Evaluating planner robustness on Out-Of-Distribution (OOD) obstacle generation.
* **2D to 3D Scaling**: Custom Gymnasium wrapper to bridge the Cross-Entropy Method (CEM) planner with the complex 3D physics of ManiSkill 3.

## 📂 Repository Structure

* `configs/` (or `conf/`): Hydra configuration files for training encoders and planning environments (`plan_pusht.yaml`, `plan_pushcube.yaml`).
* `models/`: Contains the core World Model predictor and our custom `vision_encoders.py` wrapper.
* `sim/` (or `env/`): Environment wrappers, including the custom `pushcube_wrapper.py` for ManiSkill 3 integration.
* `scripts/`: Bash scripts for SLURM cluster deployment.

## 🤖 Tasks & Environments

1. **PushT (2D)**: Precision rigid-body manipulation. The agent must push a T-shaped block into a target pose.
2. **Wall (2D)**: Zero-shot generalization test. The agent must navigate around a randomly generated wall to reach a target unseen during training.
3. **PushCube (3D)**: Real-world physics simulation using ManiSkill 3. An 8-DoF Franka Emika Panda arm pushing a cube under dynamic occlusions.

## 🧠 Evaluated Encoders

| Encoder | Pre-training Objective | Patch Spatial Locality |
| :--- | :--- | :--- |
| **DINOv2** | Self-Supervised Distillation | High |
| **CLIP** | Global Image-Text Contrastive | Low (Softmax bottleneck) |
| **SigLIP** | Pairwise Sigmoid Contrastive | High |
| **ViT-MAE** | Pixel Reconstruction (Masking) | Extremely High |

## 🛠️ Installation

Create a clean Python 3.11 environment to avoid dependency conflicts (especially with `numpy` 2.0 and `gym`):

```bash
python3 -m venv venv
source venv/bin/activate

# Install PyTorch optimized for GPUs (e.g., CUDA 12.1/12.4)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install requirements with specific stable versions
pip install numpy==1.26.4 gym==0.23.1 pymunk==6.8.0 opencv-python==4.6.0.66 \
    shapely==2.0.3 scikit-image==0.19.3 hydra-core==1.2.0 hydra-submitit-launcher==1.2.0 \
    imageio-ffmpeg==0.4.9 psutil==5.9.8 zarr==2.16.1

# Install extra dependencies for backbones and 3D simulation
pip install transformers>=4.38.0 wandb einops mani_skill
```

## 💻 Usage

### 1. Training
Train a specific backbone using the `encoder` flag. The system uses Hydra to save checkpoints in automatically categorized directories.

```bash
python train.py encoder=siglip env=pusht env.dataset.data_path=./data/pusht_noise training.batch_size=16 training.epochs=1
```

### 2. Planning (Evaluation)
Run the CEM Planner using a saved checkpoint. The outputs, including MP4 rollouts and imagined latent states, will be saved in `plan_outputs/`.

```bash
# PushT (2D)
python plan.py --config-name plan_pusht.yaml model_name=siglip/YYYY-MM-DD/HH-MM-SS n_plot_samples=10

# PushCube (3D)
python plan.py --config-name plan_pushcube.yaml model_name=dino/YYYY-MM-DD/HH-MM-SS n_plot_samples=0
```

## 📊 Results Summary

Performance was evaluated using 16 parallel simulation environments and Model Predictive Control (CEM) with identical random seeds for fair evaluation.

| Visual Backbone | PushT (Success Rate) | PushT (Mean State Dist) | Wall (Success Rate) | Wall (Mean State Dist) |
| :--- | :---: | :---: | :---: | :---: |
| **DINOv2** (Baseline) | **87.50%** | 39.95 | **87.50%** | 4.55 |
| **CLIP** (OpenAI)     | 6.25%      | 46.29 | 75.00%     | 7.08 |
| **SigLIP**            | 68.75%     | 50.42 | **87.50%** | **4.52** |
| **ViT-MAE**           | 75.00%     | **19.45** | 75.00%     | 6.81 |

*Note: ViT-MAE records the lowest spatial error on PushT, highlighting the geometric precision of reconstructive objectives. Meanwhile, SigLIP resolves CLIP's spatial degradation, matching DINOv2's zero-shot planning robustness on the Wall task.*

## 📜 Acknowledgements & References

This project builds upon the foundational research of several incredible teams. If you find this repository useful, please consider referencing the following original works:

* **DINO-WM (Base Framework):** Zhou, G., Pan, H., LeCun, Y. & Pinto, L. (2024). *DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning*. [arXiv:2411.04983](https://arxiv.org/abs/2411.04983)
* **ManiSkill 3 (3D Simulator):** Tao, S. et al. (2024). *ManiSkill: Generalizable Manipulation Skill Benchmark*. [GitHub](https://github.com/haosulab/ManiSkill) | [arXiv:2410.00425](https://arxiv.org/abs/2410.00425)
* **DINOv2:** Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
* **CLIP:** Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
* **SigLIP:** Zhai, X. et al. (2023). *Sigmoid Loss for Language Image Pre-Training*. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
* **ViT-MAE:** He, K. et al. (2022). *Masked Autoencoders Are Scalable Vision Learners*. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
* **World Models (Core Concept):** Ha, D. & Schmidhuber, J. (2018). *World Models*. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)

<p align="center">
  <img src="assets/logo.png" width="20%"/>
</p>

# NaVILA: Legged Robot Vision-Language-Action Model for Navigation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

[Website](https://navila-bot.github.io/) / [Arxiv](https://arxiv.org/abs/2412.04453) / [Huggingface](https://huggingface.co/collections/a8cheng/navila-legged-robot-vision-language-action-model-for-naviga-67cfc82b83017babdcefd4ad)

## 💡 Introduction

[**NaVILA: Legged Robot Vision-Language-Action Model for Navigation**](<>)

NaVILA is a two-level framework that combines VLAs with locomotion skills for navigation. It generates high-level language-based commands, while a real-time locomotion policy ensures obstacle avoidance.

## Evaluation

### Installation

This repository builds on [VLN-CE](https://github.com/jacobkrantz/VLN-CE), which relies on older versions of [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7) and [Habitat-Sim](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7). The installation process requires several modifications and can be complex.

To set up the environment, run:
```bash
./environment_setup_eval.sh navila-eval
```
If the installation fails, check setup.sh for potential issues. The script assumes EGL is supported in your system. If it is missing, install the required packages beforehand:
```bash
sudo apt-get update || true
sudo apt-get install -y --no-install-recommends \
     libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev
```

### Data
Please follow [VLN-CE](https://github.com/jacobkrantz/VLN-CE) and download R2R and RxR annotations, and scene data inside the `evaluation/data` folder. The data should have structure like:
```graphql
data/datasets
├─ RxR_VLNCE_v0
|   ├─ train
|   |    ├─ train_guide.json.gz
|   |    ├─ ...
|   ├─ val_unseen
|   |    ├─ val_unseen_guide.json.gz
|   |    ├─ ...
|   ├─ ...
├─ R2R_VLNCE_v1-3_preprocessed
|   ├─ train
|   |    ├─ train.json.gz
|   |    ├─ ...
|   ├─ val_unseen
|   |    ├─ val_unseen.json.gz
|   |    ├─ ...
data/scene_dataset
├─ mp3d
|   ├─ 17DRP5sb8fy
|   |    ├─ 17DRP5sb8fy.glb
|   |    ├─ ...
|   ├─ ...
```
### Run


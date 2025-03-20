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

1. Create a Conda Environment with Python 3.10
```bash
conda create -n navila-eval python=3.10
conda activate navila-eval
```

2. Build Habitat-Sim & Lab (v0.1.7) from **Source**

Follow the [VLN-CE setup guide](https://github.com/jacobkrantz/VLN-CE?tab=readme-ov-file#setup).
To resolve NumPy compatibility issues, apply the following hotfix:
```bash
python evaluation/scripts/habitat_sim_autofix.py # replace habitat_sim/utils/common.py
```

3. Install VLN-CE Dependencies
```bash
pip install -r evaluation/requirements.txt
```

4. Install VILA Dependencies
```bash
# Install FlashAttention2
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install VILA (assum in root dir)
pip install -e .
pip install -e ".[train]"
pip install -e ".[eval]"

# Install HF's Transformers
pip install git+https://github.com/huggingface/transformers@v4.37.2
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/
```

5. Fix WebDataset Version for VLN-CE Compatibility
```bash
pip install webdataset==0.1.103
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
### Running Evaluation
1. Download the checkpoint from [a8cheng/navila-llama3-8b-stage3](https://huggingface.co/a8cheng/navila-llama3-8b-stage3).
2. Run evaluation on R2R using:
```bash
cd evaluation
bash scripts/eval/r2r.sh CKPT_PATH NUM_CHUNKS CHUNK_START_IDX "GPU_IDS"
```
Examples:
* Single GPU:
    ```bash
    bash scripts/eval/r2r.sh CKPT_PATH 1 0 "0"
    ```
* Multiple GPUs (e.g., 8 GPUs):
    ```bash
    bash scripts/eval/r2r.sh CKPT_PATH 8 0 "0,1,2,3,4,5,6,7"
    ```
3. Visualized videos are saved in 
```bash
./eval_out/CKPT_NAME/VLN-CE-v1/val_unseen/videos
```
<p align="left">
  <img src="assets/sample.gif" width="600">
</p>
4. Aggregate results and view the scores

```bash
python scripts/eval_jsons.py ./eval_out/CKPT_NAME/VLN-CE-v1/val_unseen NUM_CHUNKS
```

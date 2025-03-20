<p align="center">
  <img src="assets/logo.png" width="20%"/>
</p>

# NaVILA: Legged Robot Vision-Language-Action Model for Navigation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

[Website](https://navila-bot.github.io/) / [Arxiv](https://arxiv.org/abs/2412.04453) / [Huggingface](https://huggingface.co/collections/a8cheng/navila-legged-robot-vision-language-action-model-for-naviga-67cfc82b83017babdcefd4ad)

## 💡 Introduction

[**NaVILA: Legged Robot Vision-Language-Action Model for Navigation**](<>)

NaVILA is a two-level framework that combines VLAs with locomotion skills for navigation. It generates high-level language-based commands, while a real-time locomotion policy ensures obstacle avoidance.

## Installation (Evaluation)

This repository builds on [VLN-CE](https://github.com/jacobkrantz/VLN-CE), which relies on older versions of [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7) and [Habitat-Sim](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7). The installation requires several modifications and can be somewhat complicated.

```bash
conda create -n navila-eval python=3.10 cmake==3.14.0 -y
conda activate navila-eval

cd evaluation
git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-lab.git
git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-sim.git

cd habitat-sim
git submodule update --init --recursive
python setup.py install --headless
python scripts/habitat_sim_autofix.py # auto fix np issue

cd ../habitat-lab
pip install -r requirements.txt
pip install -r habitat_baselines/rl/requirements.txt
pip install -r habitat_baselines/rl/ddppo/requirements.txt
pip install tensorflow webdataset==0.1.103
python setup.py develop --all

pip install gym==0.17.3

```

## Training

VILA training contains three steps, for specific hyperparameters, please check out the [scripts/v1_5](scripts/v1_5) folder:

### Step-1: Alignment

We utilize LLaVA-CC3M-Pretrain-595K dataset to align the textual and visual modalities.

The stage 1 script takes in two parameters and it can run on a single 8xA100 node. `BASE_MODEL_PATH` points to a online or local huggingface repository, such as `NousResearch/Llama-2-7b-hf`. `OUTPUT_NAME` points to a target directory under `checkpoints`, which will save the trained multimodal projector afterwards.

```bash
bash scripts/v1_5/paper/1_mm_align.sh [BASE_MODEL_PATH] [OUTPUT_NAME]
```

### Step-2: Pretraining

We use MMC4 and Coyo dataset to train VLM with interleaved image-text pairs.

```bash
bash scripts/v1_5/paper/2_pretrain_mmc4_coyo.sh [CODE_PATH] [BASE_MODEL_PATH] [STAGE1_PATH] [OUTPUT_NAME]
```

The stage 2 script takes in four arguments. `CODE_PATH` is the absolute path to our VILA codebase, `BASE_MODEL_PATH` has similar meaning to what is presented in the stage 1 script. `STAGE1_PATH` points to the `OUTPUT_NAME` of stage 1 (i.e. where the stage 1 checkpoint is stored). `OUTPUT_NAME` is the desired folder name under `checkpoints` that saves the pretraining checkpoint. The script we provided for this stage is executed on slurm, and we expect it to execute on 16 nodes (128 GPUs).

### Step-3: Supervised fine-tuning

This is the last stage of VILA training, in which we tune the model to follow multimodal instructions on a subset of M3IT, FLAN and ShareGPT4V. This stage runs on a 8xA100 node.

```bash
bash scripts/v1_5/paper/3_sft.sh [STAGE2_PATH] [OUTPUT_NAME]
```

The stage 3 script takes in two arguments. `STAGE2_PATH` points to the `OUTPUT_NAME` of the stage 2 script (i.e. where the stage 2 checkpoint is stored). `OUTPUT_NAME` is the desired folder name under `checkpoints` that stores the final checkpoint.

## Evaluations

### Image Benchmarks

You can follow [Llava1.5 eval](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) to download all datasets. After downloading all datasets, please put them under `playground/data/eval`.

Please make the following changes to the MME evaluation script. Please search for:

```python
data_path = "MME_Benchmark_release_version"
```

and replace it with:

```python
data_path = os.path.join(script_dir, "MME_Benchmark_release_version")
```

We provide a push-the-button script to perform evaluation on all 10 datasets that do not require GPT-assisted evaluation:

```bash
./scripts/v1_5/eval/eval_all.sh [CHECKPOINT_PATH] [MODEL_NAME] [CONV_MODE]
```

This script takes in two parameters, `CHECKPOINT_PATH` points to the stage 3 model checkpoint, and `MODEL_NAME` will be the name of evaluation results.

[VQAv2](https://eval.ai/web/challenges/challenge-page/830/my-submission) and [Vizwiz](https://eval.ai/web/challenges/challenge-page/2185/my-submission) evaluations are hosted on eval.ai. You need to register an account and create a team to be able to submit eval.

MMBench and MMBench_CN eval are hosted on another [evaluation server](https://opencompass.org.cn/leaderboard-multimodal). Make sure you change the name of the file before submitting, otherwise the server caches results and will always return wrong result to you.

We provide a quick script to automatically organize the prediction files that need to be submitted to servers:

```bash
python scripts/v1_5/eval/copy_predictions.py [MODEL_NAME]
```

You will be able to find the predictions under `playground/data/predictions_upload/[MODEL_NAME]` after executing this script.

### Video Benchmarks

Please follow the evaluation steps in [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md#data-for-validating) for dataset preparation.

```bash
./scripts/v1_5/eval/video_chatgpt/run_all.sh [CHECKPOINT_PATH] [MODEL_NAME] [CONV_MODE]
./scripts/v1_5/eval/video_chatgpt/eval_all.sh [MODEL_NAME]
```

## Inference

We provide snippets for quick inference with user prompts and images.

Llama-3-VILA1.5-8B inference:

```bash
python -W ignore llava/eval/run_vila.py \
    --model-path Efficient-Large-Model/Llama-3-VILA1.5-8b \
    --conv-mode llama_3 \
    --query "<image>\n Please describe the traffic condition." \
    --image-file "av.png"
```

VILA1.5-40B inference:

```bash
python -W ignore llava/eval/run_vila.py \
    --model-path Efficient-Large-Model/VILA1.5-40b \
    --conv-mode hermes-2 \
    --query "<image>\n Please describe the traffic condition." \
    --image-file "av.png"
```

VILA1.5-3B video inference:

```bash
python -W ignore llava/eval/run_vila.py \
    --model-path Efficient-Large-Model/VILA1.5-3b \
    --conv-mode vicuna_v1 \
    --query "<video>\n Please describe this video." \
    --video-file "demo.mp4"
```

## Quantization and Deployment

Our VILA models are quantized by [AWQ](https://arxiv.org/abs/2306.00978) into 4 bits for efficient inference on the edge. We provide a push-the-button [script](https://github.com/mit-han-lab/llm-awq/blob/main/scripts/vila_example.sh) to quantize VILA with AWQ.

### Running VILA on desktop GPUs and edge GPUs

We support AWQ-quantized 4bit VILA on GPU platforms via [TinyChat](https://github.com/mit-han-lab/llm-awq/tree/main/tinychat). We provide a [tutorial](https://github.com/mit-han-lab/llm-awq/tree/main/tinychat#support-vlm-models-vila--llava) to run the model with TinyChat after quantization. We also provide an [instruction](https://github.com/mit-han-lab/llm-awq/tree/main/tinychat/serve) to launch a Gradio server (powered by TinyChat and AWQ) to serve 4-bit quantized VILA models.

### Running VILA on laptops

We further support our AWQ-quantized 4bit VILA models on various CPU platforms with both x86 and ARM architectures with our [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine). We also provide a detailed [tutorial](https://github.com/mit-han-lab/TinyChatEngine/tree/main?tab=readme-ov-file#deploy-vision-language-model-vlm-chatbot-with-tinychatengine) to help the users deploy VILA on different CPUs.

### Running VILA API server

A simple API server has been provided to serve VILA models. The server is built on top of [FastAPI](https://fastapi.tiangolo.com/) and [Huggingface Transformers](https://huggingface.co/transformers/). The server can be run with the following command:

#### With CLI

```bash
python -W ignore server.py \
    --port 8000 \
    --model-path Efficient-Large-Model/VILA1.5-3B \
    --conv-mode vicuna_v1
```

#### With Docker

```bash
docker build -t vila-server:latest .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ./hub:/root/.cache/huggingface/hub \
    -it --rm -p 8000:8000 \
    -e VILA_MODEL_PATH=Efficient-Large-Model/VILA1.5-3B \
    -e VILA_CONV_MODE=vicuna_v1 \
    vila-server:latest
```

Then you can call the endpoint with the OpenAI SDK as follows:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000",
    api_key="fake-key",
)
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://blog.logomyway.com/wp-content/uploads/2022/01/NVIDIA-logo.jpg",
                        # Or you can pass in a base64 encoded image
                        # "url": "data:image/png;base64,<base64_encoded_image>",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
    model="VILA1.5-3B",
    # You can pass in extra parameters as follows
    extra_body={"num_beams": 1, "use_cache": False},
)
print(response.choices[0].message.content)
```

<sup>NOTE: This API server is intended for evaluation purposes only and has not been optimized for production use. It has only been tested on A100 and H100 GPUs.</sup>

## Checkpoints

We release [VILA1.5-3B](https://hf.co/Efficient-Large-Model/VILA1.5-3b), [VILA1.5-3B-S2](https://hf.co/Efficient-Large-Model/VILA1.5-3b-s2), [Llama-3-VILA1.5-8B](https://hf.co/Efficient-Large-Model/Llama-3-VILA1.5-8b), [VILA1.5-13B](https://hf.co/Efficient-Large-Model/VILA1.5-13b), [VILA1.5-40B](https://hf.co/Efficient-Large-Model/VILA1.5-40b) and the 4-bit [AWQ](https://arxiv.org/abs/2306.00978)-quantized models [VILA1.5-3B-AWQ](https://hf.co/Efficient-Large-Model/VILA1.5-3b-AWQ), [VILA1.5-3B-S2-AWQ](https://hf.co/Efficient-Large-Model/VILA1.5-3b-s2-AWQ), [Llama-3-VILA1.5-8B-AWQ](https://hf.co/Efficient-Large-Model/Llama-3-VILA1.5-8b-AWQ), [VILA1.5-13B-AWQ](https://hf.co/Efficient-Large-Model/VILA1.5-13b-AWQ), [VILA1.5-40B-AWQ](https://hf.co/Efficient-Large-Model/VILA1.5-40b-AWQ).

## 🔒 License

- The code is released under the Apache 2.0 license as found in the [LICENSE](./LICENSE) file.
- The pretrained weights are released under the [CC-BY-NC-SA-4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).
- The service is a research preview intended for non-commercial use only, and is subject to the following licenses and terms:
  - [Model License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA. For LLAMA3-VILA checkpoints terms of use, please refer to the [LLAMA3 License](https://llama.meta.com/llama3/license/) for additional details.
  - [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI
  - [Dataset Licenses](./data_prepare/LICENSE) for each one used during training.

## Team

| | | |
| --- | --- | ---|
[\*Yao Lu](https://scholar.google.com/citations?user=OI7zFmwAAAAJ&hl=en): Nvidia|  [\*Hongxu Yin](https://hongxu-yin.github.io/): Nvidia |  [\*Ji Lin](https://www.linji.me/): OpenAI (work done at Nvidia and MIT)
[Wei Ping](https://scholar.google.com/citations?user=6gKEYRgAAAAJ&hl=en): Nvidia |   [Pavlo Molchanov](https://www.pmolchanov.com/): Nvidia |  [Andrew Tao](https://scholar.google.com/citations?user=Wel9l1wAAAAJ&hl=en): Nvidia |
[Haotian Tang](http://kentang.net/): MIT |  [Shang Yang](https://ys-2020.github.io/): MIT |  [Ligeng Zhu](https://lzhu.me/): Nvidia, MIT |
[Wei-Chen Wang](https://weichenwang.me/): MIT |  [Fuzhao Xue](https://xuefuzhao.github.io/): Nvidia, NUS |  [Yunhao Fang](https://seerkfang.github.io/): Nvidia, UCSD |
[Yukang Chen](https://yukangchen.com/): Nvidia, CUHK | [Zhuoyang Zhang](https://openreview.net/profile?id=~Zhuoyang_Zhang1): Nvidia, Tsinghua Univ. | [Yue Shen](https://www.linkedin.com/in/yue-james-shen/): Nvidia |
[Wei-Ming Chen](https://scholar.google.com/citations?user=6xFvyJwAAAAJ&hl=en): Nvidia |  [Huizi Mao](https://scholar.google.com/citations?user=r5WezOYAAAAJ&hl=zh-CN): Nvidia | [Baifeng Shi](https://bfshi.github.io/): Nvidia, UC Berkeley |
[Jan Kautz](https://jankautz.com/): Nvidia | [Mohammad Shoeybi](https://scholar.google.com/citations?user=62ElavIAAAAJ&hl=en): Nvidia | [Song Han](http://songhan.mit.edu/): Nvidia, MIT

## Citations

```
@misc{lin2023vila,
      title={VILA: On Pre-training for Visual Language Models},
      author={Ji Lin and Hongxu Yin and Wei Ping and Yao Lu and Pavlo Molchanov and Andrew Tao and Huizi Mao and Jan Kautz and Mohammad Shoeybi and Song Han},
      year={2023},
      eprint={2312.07533},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for their wonderful work.
- [InternVL](https://github.com/OpenGVLab/InternVL): for open-sourcing InternViT (used in VILA1.5-40b) and the [InternVL-SFT](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat#prepare-training-datasets) data blend (inspired by LLaVA-1.6) used in all VILA1.5 models.
- [Vicuna](https://github.com/lm-sys/FastChat): the amazing open-sourced large language model!
- [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT): we borrowed video evaluation script from this repository.
- [MMC4](https://github.com/allenai/mmc4), [COYO-700M](https://github.com/kakaobrain/coyo-dataset), [M3IT](https://huggingface.co/datasets/MMInstruction/M3IT), [OpenORCA/FLAN](https://huggingface.co/datasets/Open-Orca/FLAN), [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V), [WIT](google-research-datasets/wit), [GSM8K-ScRel](https://github.com/OFA-Sys/gsm8k-ScRel/blob/main/data/train_use.jsonl), [VisualGenome](https://visualgenome.org/api/v0/api_home.html), [VCR](https://visualcommonsense.com/download/), [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA), [Shot2Story](https://github.com/bytedance/Shot2Story/blob/master/DATA.md), [Youcook2](http://youcook2.eecs.umich.edu/), [Vatex](https://eric-xw.github.io/vatex-website/download.html), [ShareGPT-Video](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction) for providing datasets used in this research.

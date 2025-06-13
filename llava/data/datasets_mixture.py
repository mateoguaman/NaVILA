# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import dataclass, field


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    meta_path: str = field(default=None, metadata={"help": "Path to the meta data for webdataset."})
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})
    caption_choice: str = field(default=None, metadata={"help": "Path to the caption directory for recaption."})
    description: str = field(
        default=None,
        metadata={
            "help": "Detailed desciption of where the data is from, how it is labelled, intended use case and the size of the dataset."
        },
    )
    test_script: str = (None,)
    maintainer: str = (None,)
    ############## ############## ############## ############## ############## ##############
    caption_choice: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    caption_choice_2: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    start_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})
    end_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})


DATASETS_LEGACY = {}


def add_dataset(dataset):
    if dataset.dataset_name in DATASETS_LEGACY:
        # make sure the data_name is unique
        warnings.warn(f"{dataset.dataset_name} already existed in DATASETS. Make sure the name is unique.")
    assert "+" not in dataset.dataset_name, "Dataset name cannot include symbol '+'."
    DATASETS_LEGACY.update({dataset.dataset_name: dataset})


def register_datasets_mixtures():

    llava_1_5_mm_align = Dataset(
        dataset_name="llava_1_5_mm_align",
        dataset_type="torch",
        data_path="/PATH_TO_DATA/LLaVA-CC3M-Pretrain-595K/chat.json",
        image_path="/PATH_TO_DATA/LLaVA-CC3M-Pretrain-595K/images",
    )
    add_dataset(llava_1_5_mm_align)

    video_chatgpt = Dataset(
        dataset_name="video_chatgpt",
        dataset_type="torch",
        data_path="/PATH_TO_DATA/annotations.json",
        image_path="/PATH_TO_DATA/videos",
    )
    add_dataset(video_chatgpt)

    sharegpt_video = Dataset(
        dataset_name="sharegpt_video",
        dataset_type="torch",
        data_path="/PATH_TO_DATA/annotations.json",
        image_path="/PATH_TO_DATA/videos",
    )
    add_dataset(sharegpt_video)

    sharegpt4v_sft = Dataset(
        dataset_name="sharegpt4v_sft",
        dataset_type="torch",
        data_path="/PATH_TO_DATA/annotations.json",
        image_path="/PATH_TO_DATA/images",
    )
    add_dataset(sharegpt4v_sft)

    r2r_aug = Dataset(
        dataset_name="r2r_aug",
        dataset_type="vlnce",
        data_path="/PATH_TO_DATA/R2R/v5/annotations_aug.json",
        image_path="/PATH_TO_DATA/R2R/v5/videos",
        description="350K VLN-CE R2R data. (augmented aith duplicate samples)",
    )
    add_dataset(vlnce_v5_aug)

    envdrop = Dataset(
        dataset_name="envdrop",
        dataset_type="envdrop",
        data_path="/PATH_TO_DATA/R2R/envdrop/annotations.json",
        image_path="/PATH_TO_DATA/R2R/envdrop/videos",
        description="VLN_CE Envdrop.",
    )
    add_dataset(envdrop)

    scanqa = Dataset(
        dataset_name="scanqa",
        dataset_type="torch",
        data_path="/PATH_TO_DATA/ScanQA/annotations/ScanQA_v1.0_train_reformat.json",
        image_path="/PATH_TO_DATA/ScanQA/videos",
        description="ScanQA training set.",
    )
    add_dataset(scanqa)

    rxr_aug = Dataset(
        dataset_name="rxr_aug",
        dataset_type="vlnce",
        data_path="/PATH_TO_DATA/RxR/v5/annotations_aug.json",
        image_path="/PATH_TO_DATA/RxR/v5/videos",
        description="400K RxR data. (augmented aith duplicate stops only - 5x)",
    )
    add_dataset(rxr_aug)

    real_aug = Dataset(
        dataset_name="real_aug",
        dataset_type="vlnce",
        data_path="/PATH_TO_DATA/real/annotations_aug.json",
        image_path="/PATH_TO_DATA/real/videos",
        description="560K Real augmented, no direction is included. (augmented aith duplicate stops only - 5x)",
    )
    add_dataset(real_aug)

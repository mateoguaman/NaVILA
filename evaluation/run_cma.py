import argparse
import os
import random

import numpy as np
import torch
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from vlnce_baselines.config.default import get_config


def initialize_policy_from_ckpt(config: Config, ckpt_path: str):

    env = get_env_class(config.ENV_NAME)(config=config)

    baseline_policy = baseline_registry.get_policy(config.MODEL.policy_name)
    policy = baseline_policy.from_config(
        config=config,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    ckpt_dict = torch.load(ckpt_path, map_location="cpu")
    policy.load_state_dict(ckpt_dict["state_dict"])
    logger.info(f"Loaded weights from checkpoint: {ckpt_path}")

    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=False,
        default="vlnce_baselines/config/r2r_baselines/my_cma.yaml",
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, opts=None) -> None:
    config = get_config(exp_config, opts)

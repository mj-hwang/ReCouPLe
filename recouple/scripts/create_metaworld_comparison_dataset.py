import argparse
import os

import gym
import numpy as np
import torch
import random

import research
from research.datasets.ipl_dataset import ReplayBuffer, PairwiseComparisonDataset

TRAJ_LEN = 64
LANG_MODEL_NAME = "google-t5/t5-small"

if __name__ == "__main__":
    # This is a short script that generates a pairwise preference dataset from a ReplayBuffer

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Env associated with the dataset.")
    parser.add_argument("--path", type=str, required=True, help="Path to the ReplayBuffer")
    parser.add_argument("--output", type=str, required=True, help="Output path for the dataset")
    parser.add_argument("--capacity", type=int, default=4000, help="How big to make the dataset")
    parser.add_argument("--segment-size", type=int, default=32, help="How large to make segments")
    args = parser.parse_args()

    assert os.path.exists(args.path)

    # TODO: support D4RL datasets

    env = gym.make("mw_" + args.env)
    capacity = args.capacity  # Rename for ease

    # Set to not distributed so we immediately load all of the data
    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, distributed=False, path=args.path, sample_multiplier=2.0,
    )

    batch = replay_buffer.sample(batch_size=2 * capacity, stack=args.segment_size, pad=0,    )
    returns = np.sum(batch["reward"], axis=1)
    labels = 1.0 * (returns[:capacity] < returns[capacity:])    

    np.savez_compressed(
        f"comparison_dataset/data_mw_single_task_{args.env}.npz", 
        obs_1=batch["obs"][:capacity], 
        action_1=batch["action"][:capacity], 
        obs_2=batch["obs"][capacity:], 
        action_2=batch["action"][capacity:], 
        label=labels,
    )


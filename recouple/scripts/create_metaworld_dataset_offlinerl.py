import argparse
import os
import torch

import gym
import numpy as np

import research
from research.datasets.ipl_dataset import ReplayBuffer, PairwiseComparisonDataset

from transformers import AutoTokenizer, T5Config, T5EncoderModel

TASK_TEXTS = {
    "pick-place-v2": "pick and place a puck to a goal",
    "pick-place-wall-v2": "pick a puck, bypass a wall and place the puck to a goal",
    "push-v2": "push a puck to a goal",
    "push-wall-v2": "bypass a wall and push a puck to a goal",
    "reach-v2": "reach a goal position",
    "reach-wall-v2": "bypass a wall and reach a goal position",
}

def create_reward_label(features_diff):
    # This function creates a reward label for the dataset
    pass

TRAJ_LEN = 64
LANG_MODEL_NAME = "google-t5/t5-small"

if __name__ == "__main__":
    # This is a short script that generates a pairwise preference dataset from a ReplayBuffer

    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=str, nargs="+", required=True, help="Envs associated with the dataset.")
    parser.add_argument("--paths", type=str, nargs="+", required=True, help="Path to the ReplayBuffer")
    parser.add_argument("--output", type=str, required=True, help="Output path for the dataset")
    parser.add_argument("--capacity", type=int, default=10000, help="How big to make the dataset for each env")
    parser.add_argument("--segment-size", type=int, default=64, help="How large to make segments")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    capacity = args.capacity

    tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL_NAME)
    lang_encoder = T5EncoderModel.from_pretrained(LANG_MODEL_NAME)
    lang_encoder.config.dropout_rate = 0.0
    lang_encoder.eval()
    for param in lang_encoder.parameters():
        param.requires_grad = False

    task_texts = [TASK_TEXTS[env] for env in args.envs]
    tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL_NAME)
    tokenized_tasks = tokenizer(
        task_texts,
        padding=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    task_tokens = tokenized_tasks["input_ids"]
    task_masks = tokenized_tasks["attention_mask"]
    task_lhs = lang_encoder(task_tokens, attention_mask=task_masks).last_hidden_state
    float_task_mask = task_masks.unsqueeze(-1).type_as(task_lhs)
    masked_task_lhs = task_lhs * float_task_mask
    task_embeddings = masked_task_lhs.sum(dim=1) / float_task_mask.sum(dim=1)  
    
    combined_obs = []
    combined_next_obs = []
    combined_action = []
    combined_reward = []
    combined_task_idxs = []

    np.random.seed(args.seed)

    for env_name, path in zip(args.envs, args.paths):
        print(f"Creating dataset for {env_name} at {path}")
        assert os.path.exists(path), f"Path {path} does not exist"

        env = gym.make("mw_" + env_name)
          # Rename for ease

        # Set to not distributed so we immediately load all of the data
        replay_buffer = ReplayBuffer(
            env.observation_space, env.action_space, distributed=False, path=path, sample_multiplier=2.0,
        )

        batch = replay_buffer.sample(capacity, pad=0)
        combined_obs.append(batch["obs"])
        combined_next_obs.append(batch["next_obs"])
        combined_action.append(batch["action"])
        combined_reward.append(batch["reward"] / 10.0)

    combined_obs = np.concatenate(combined_obs, axis=0)
    combined_next_obs = np.concatenate(combined_next_obs, axis=0)
    combined_action = np.concatenate(combined_action, axis=0)
    combined_reward = np.concatenate(combined_reward, axis=0)
    
    combined_task_embeddings = torch.repeat_interleave(task_embeddings, repeats=capacity, dim=0)
    combined_task_embeddings = combined_task_embeddings.numpy()

    print("obs:", combined_obs.shape)
    print("next_obs:", combined_next_obs.shape)
    print("action:", combined_action.shape)
    print("reward:", combined_reward.shape)
    print("task_embeddings:", combined_task_embeddings.shape)

    np.savez_compressed(
        f"offline_dataset/data_offlinerl_{'_'.join(args.envs)}_gt_seed_{args.seed}_{args.capacity}.npz", 
        obs=combined_obs, 
        next_obs=combined_next_obs, 
        action=combined_action,
        reward=combined_reward,
        task_embeddings=combined_task_embeddings,
    )

import argparse
import os
import torch

import gym
import numpy as np

import research
from research.datasets.ipl_dataset import ReplayBuffer, PairwiseComparisonDataset

from transformers import AutoTokenizer, T5Config, T5EncoderModel

# TASK_TEXTS = {
#     "pick-place-v2": "pick and place a puck to a goal",
#     "pick-place-wall-v2": "pick a puck, bypass a wall and place the puck to a goal",
#     "push-v2": "push a puck to a goal",
#     "push-wall-v2": "bypass a wall and push a puck to a goal",
#     "reach-v2": "reach a goal position",
#     "reach-wall-v2": "bypass a wall and reach a goal position",
# }

TASK_TEXTS = {
    "pick-place-v2":       "pick up puck, lift it, and place it on goal",
    "pick-place-wall-v2":  "pick up puck, use waypoint to bypass wall, and place it on goal",
    "push-v2":             "make contact and push puck to goal",
    "push-wall-v2":        "make contact, bypass wall via waypoint, and push puck to goal",
    "reach-v2":            "move gripper to goal position",
    "reach-wall-v2":       "move gripper to goal position, bypassing wall",
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
    parser.add_argument("--capacity", type=int, default=100, help="How big to make the dataset for each env")
    parser.add_argument("--segment-size", type=int, default=16, help="How large to make segments")
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
    
    combined_obs_1 = []
    combined_action_1 = []
    combined_obs_2 = []
    combined_action_2 = []
    combined_labels = []
    combined_task_idxs = []

    for env_name, path in zip(args.envs, args.paths):
        print(f"Creating dataset for {env_name} at {path}")
        assert os.path.exists(path), f"Path {path} does not exist"

        env = gym.make("mw_" + env_name)
          # Rename for ease

        # Set to not distributed so we immediately load all of the data
        replay_buffer = ReplayBuffer(
            env.observation_space, env.action_space, distributed=False, path=path, sample_multiplier=2.0,
        )

        batch = replay_buffer.sample(batch_size=2 * capacity, stack=args.segment_size, pad=0)
        returns = np.sum(batch["reward"], axis=1)
        queries = dict(
            obs_1=batch["obs"][:capacity],
            obs_2=batch["obs"][capacity:],
            action_1=batch["action"][:capacity],
            action_2=batch["action"][capacity:],
        )
        labels = 1.0 * (returns[:capacity] < returns[capacity:])

        combined_obs_1.append(batch["obs"][:capacity])
        combined_action_1.append(batch["action"][:capacity])
        combined_obs_2.append(batch["obs"][capacity:])
        combined_action_2.append(batch["action"][capacity:])
        combined_labels.append(labels)

    combined_obs_1 = np.concatenate(combined_obs_1, axis=0)
    combined_action_1 = np.concatenate(combined_action_1, axis=0)
    combined_obs_2 = np.concatenate(combined_obs_2, axis=0)
    combined_action_2 = np.concatenate(combined_action_2, axis=0)
    combined_labels = np.concatenate(combined_labels, axis=0)
    
    combined_task_embeddings = torch.repeat_interleave(task_embeddings, repeats=capacity, dim=0)
    combined_task_embeddings = combined_task_embeddings.numpy()
    task_labels = np.repeat(np.arange(len(args.envs)), capacity)
    print(task_labels)

    print("obs_1:", combined_obs_1.shape)
    print("action_1:", combined_action_1.shape)
    print("obs_2:", combined_obs_2.shape)
    print("action_2:", combined_action_2.shape)
    print("label:", combined_labels.shape)
    print("task_embeddings:", combined_task_embeddings.shape)
    print("task_label:", task_labels.shape)

    np.savez_compressed(
        f"comparison_dataset/data_mw_multitask_with_reason_validation.npz", 
        obs_1=combined_obs_1, 
        action_1=combined_action_1, 
        obs_2=combined_obs_2, 
        action_2=combined_action_2, 
        label=combined_labels,
        task_embeddings=combined_task_embeddings,
        task_labels=task_labels,
    )

import argparse
import os
import torch

import gym
import numpy as np
import torch
import random

import research
from research.datasets.ipl_dataset import ReplayBuffer, PairwiseComparisonDataset

from transformers import AutoTokenizer, T5EncoderModel

REASON_FEATURE_DICT = {
    "pick-place-v2": ["H_GG", "L", "I_goal_L", "S"],
    "pick-place-wall-v2": ["H_GM", "L", "I_mid_L_noPW", "L_PW", "I_goal_L_PW", "S"],
    "push-v2": ["G", "T", "G_T", "I_goal_T", "S"],
    "push-wall-v2": ["G", "T", "I_mid_T", "T_PW", "I_goal_T_PW", "S"],
    "reach-v2": ["IG"],
}

REASON_FEATURES = [
    "H_GG",
    "H_GM",
    "G",
    "T",
    "G_T",
    "I_mid_T",
    "I_goal_T",
    "T_PW",
    "I_goal_T_PW",
    "L",
    "I_goal_L",
    "I_mid_L_noPW",
    "L_PW",
    "I_goal_L_PW",
    "IG",
    "S",
]

REASON_TEXT = {
    "H_GG":               "keeps firm grip while moving puck toward goal",
    "H_GM":               "keeps firm grip while moving puck toward waypoint",
    "G":                  "maintains firm grip on puck",
    "T":                  "makes contact with puck sooner",
    "G_T":                "pushes more decisively after making contact",
    "I_mid_T":            "pushes puck toward waypoint",
    "I_goal_T":           "pushes puck closer to goal",
    "T_PW":               "guides puck past wall",
    "I_goal_T_PW":        "pushes puck toward goal after clearing wall",
    "L":                  "lifts puck cleanly",
    "I_goal_L":           "carries puck toward goal while lifted",
    "I_mid_L_noPW":       "carries puck toward waypoint while lifted",
    "L_PW":               "clears wall with puck lifted",
    "I_goal_L_PW":        "carries puck toward goal after clearing wall",
    "IG":                 "moves gripper closer to goal position",
    "S":                  "finishes at goal spot",
}

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

def set_seed(seed_value):
    """Sets a global seed for reproducibility across all libraries."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    
    # These lines are crucial for CUDA operations to be deterministic.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Global seed set to {seed_value} for full reproducibility.")

TRAJ_LEN = 64
LANG_MODEL_NAME = "google-t5/t5-small"

if __name__ == "__main__":
    # This is a short script that generates a pairwise preference dataset from a ReplayBuffer

    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=str, nargs="+", required=True, help="Envs associated with the dataset.")
    parser.add_argument("--paths", type=str, nargs="+", required=True, help="Path to the ReplayBuffer")
    parser.add_argument("--output", type=str, required=True, help="Output path for the dataset")
    parser.add_argument("--capacity", type=int, default=2000, help="How big to make the dataset for each env")
    parser.add_argument("--segment-size", type=int, default=16, help="How large to make segments")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    set_seed(args.seed)

    capacity = args.capacity
    tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL_NAME)
    lang_encoder = T5EncoderModel.from_pretrained(LANG_MODEL_NAME)
    lang_encoder.config.dropout_rate = 0.0
    lang_encoder.eval()
    for param in lang_encoder.parameters():
        param.requires_grad = False

    reason_texts = [REASON_TEXTS[feature] for feature in REASON_FEATURES]
    tokenized_reasons = tokenizer(
        reason_texts,
        padding=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    reason_tokens = tokenized_reasons["input_ids"]
    reason_masks = tokenized_reasons["attention_mask"]
    reason_lhs = lang_encoder(reason_tokens, attention_mask=reason_masks).last_hidden_state
    float_reason_mask = reason_masks.unsqueeze(-1).type_as(reason_lhs)
    masked_reason_lhs = reason_lhs * float_reason_mask
    reason_embeddings = masked_reason_lhs.sum(dim=1) / float_reason_mask.sum(dim=1)  

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
    combined_reason_idxs = []
    combined_task_idxs = []

    for i, (env_name, path) in enumerate(zip(args.envs, args.paths)):
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

        features = np.sum(batch["feature"], axis=1)
        features_diff = (features[capacity:] - features[:capacity]) * (2*labels[:, np.newaxis]-1)
        softmax_features_diff = np.exp(features_diff) / np.exp(features_diff).sum(axis=1, keepdims=True)
        
        reason_idxs = [REASON_FEATURES.index(feature) for feature in REASON_FEATURE_DICT[env_name]]
        sampled_reason_idxs = np.array([np.random.choice(reason_idxs, p=row) for row in softmax_features_diff])
        current_task_embeddings = torch.repeat_interleave(task_embeddings[[i]], repeats=capacity*args.segment_size*2, dim=0)

        np.savez_compressed(
            f"comparison_dataset/data_mw_singletask_{env_name}_{args.seed}.npz", 
            obs_1=batch["obs"][:capacity], 
            action_1=batch["action"][:capacity], 
            obs_2=batch["obs"][capacity:], 
            action_2=batch["action"][capacity:], 
            label=labels,
        )

        np.savez_compressed(
            f"comparison_dataset/data_mw_offlinerl_{env_name}_{args.seed}.npz", 
            obs=batch["obs"].reshape(-1, batch["obs"].shape[-1]), 
            next_obs=batch["next_obs"].reshape(-1, batch["next_obs"].shape[-1]), 
            action=batch["action"].reshape(-1, batch["action"].shape[-1]),
            reward=batch["reward"].flatten(),
            task_embeddings=current_task_embeddings.numpy(),
        )

        combined_obs_1.append(batch["obs"][:capacity])
        combined_action_1.append(batch["action"][:capacity])
        combined_obs_2.append(batch["obs"][capacity:])
        combined_action_2.append(batch["action"][capacity:])
        combined_labels.append(labels)
        combined_reason_idxs.append(sampled_reason_idxs)

    combined_obs_1 = np.concatenate(combined_obs_1, axis=0)
    combined_action_1 = np.concatenate(combined_action_1, axis=0)
    combined_obs_2 = np.concatenate(combined_obs_2, axis=0)
    combined_action_2 = np.concatenate(combined_action_2, axis=0)
    combined_labels = np.concatenate(combined_labels, axis=0)
    
    combined_reason_idxs = np.concatenate(combined_reason_idxs, axis=0)
    combined_reason_idxs = torch.from_numpy(combined_reason_idxs).long()

    combined_reason_embeddings = reason_embeddings[combined_reason_idxs]
    combined_task_embeddings = torch.repeat_interleave(task_embeddings, repeats=capacity, dim=0)

    combined_reason_embeddings = combined_reason_embeddings.numpy()
    combined_task_embeddings = combined_task_embeddings.numpy()

    task_labels = np.repeat(np.arange(len(args.envs)), capacity)

    print("obs_1:", combined_obs_1.shape)
    print("action_1:", combined_action_1.shape)
    print("obs_2:", combined_obs_2.shape)
    print("action_2:", combined_action_2.shape)
    print("label:", combined_labels.shape)
    print("reason_embeddings:", combined_reason_embeddings.shape)
    print("task_embeddings:", combined_task_embeddings.shape)
    print("task_label:", task_labels.shape)

    np.savez_compressed(
        f"comparison_dataset/data_mw_multitask_with_reason_{args.seed}.npz", 
        obs_1=combined_obs_1, 
        action_1=combined_action_1, 
        obs_2=combined_obs_2, 
        action_2=combined_action_2, 
        label=combined_labels,
        reason_embeddings=combined_reason_embeddings,
        task_embeddings=combined_task_embeddings,
        task_labels=task_labels,
    )

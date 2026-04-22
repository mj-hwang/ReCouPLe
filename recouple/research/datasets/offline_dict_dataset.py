import math
from typing import Optional

import gym
import pickle
import numpy as np
import torch


class OfflineDictDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        path: Optional[str] = None,
        discount: float = 0.99,
        action_eps: float = 1e-5,
        segment_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        capacity: Optional[int] = None,
        mode: str = "comparison",
        label_key: str = "label",
        reward_scale: float = 1.0,
        reward_shift: float = 0.0,
        normalize_reward: bool = True,
    ):
        if path.endswith(".npz"):
            self._dataset = np.load(path)
        else:
            raise ValueError(f"Unsupported file extension: {path}")
        self.dataset = {}
        for key in self._dataset.files:
            print("loading", key)
            if key == "reward" and normalize_reward:
                min_reward = np.min(self._dataset["reward"])
                max_reward = np.max(self._dataset["reward"])
                self.dataset[key] = (reward_scale * (self._dataset["reward"] - min_reward) / (max_reward - min_reward)) + reward_shift
            else:
                self.dataset[key] = torch.from_numpy(self._dataset[key]) # .to("cuda")
            
    def __len__(self):
        return len(self.dataset["action"])
    
    def __getitem__(self, idx):
        output = {}
        for key in self.dataset.keys():
            output[key] = self.dataset[key][idx]
        return output
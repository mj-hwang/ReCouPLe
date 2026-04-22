import random
from typing import Optional

import h5py
import numpy as np
import torch

from research.utils import utils

from .replay_buffer.buffer import ReplayBuffer


class ManiSkillDataset(ReplayBuffer):
    """
    Simple Class that writes the data from the ManiSkillDatasets into a ReplayBuffer
    """

    def __init__(
        self, observation_space, action_space, *args, action_eps: Optional[float] = 0.0, train=True, **kwargs
    ):
        self.action_eps = action_eps
        self.train = train
        super().__init__(observation_space, action_space, *args, **kwargs)

    def _data_generator(self):
        # Compute the worker info
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id

        f = h5py.File(self.path, "r")

        # Assign demos to each worker
        demos = list(f.keys())  # Deterministic ordering
        demos = demos[worker_id::num_workers]
        # Shuffle the data ordering
        random.shuffle(demos)

        for _i, demo in enumerate(demos):
            # Get obs from the start to the end.
            obs = f[demo]["obs"]["sensor_data"]["base_camera"]["rgb"][:]
            obs = np.transpose(obs , (0, 3, 1, 2))
            obs = utils.remove_float64(obs)

            dummy_action = np.expand_dims(self.dummy_action, axis=0)
            # breakpoint()
            action = np.concatenate((dummy_action, f[demo]["actions"][:]), axis=0)
            action = utils.remove_float64(action)

            if self.action_eps is not None:
                lim = 1 - self.action_eps
                action = np.clip(action, -lim, lim)

            reward = np.concatenate(([0], f[demo]["rewards"][:]), axis=0)
            reward = utils.remove_float64(reward)

            done = np.concatenate(([0], f[demo]["terminated"][:]), axis=0).astype(np.bool_)
            done[-1] = True

            discount = (1 - done).astype(np.float32)

            yield dict(obs=obs, action=action, reward=reward, done=done, discount=discount)

        f.close()  # Close the file handler.

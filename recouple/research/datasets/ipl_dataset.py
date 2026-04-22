import copy
import datetime
import io
import os
import random
import shutil
import tempfile
from typing import Any, Callable, Dict, Optional, Union

import gym
import numpy as np
import torch

from research.utils import utils


def save_data(data: Dict, path: str) -> None:
    # Perform checks to make sure everything needed is in the data object
    assert all([k in data for k in ("obs", "action", "reward", "done", "feature")])
    # Flatten everything for saving as an np array
    data = utils.flatten_dict(data)
    # Format everything into numpy in case it was saved as a list
    for k in data.keys():
        if not isinstance(data[k], np.ndarray):
            assert isinstance(data[k], list), "Unknown type passed to save_data"
            first_value = data[k][0]
            if isinstance(first_value, (np.float64, float)):
                dtype = np.float32  # Detect and convert out float64
            elif isinstance(first_value, (np.ndarray, np.generic)):
                dtype = first_value.dtype
            elif isinstance(first_value, int):
                dtype = np.int64
            elif isinstance(first_value, bool):
                dtype = np.bool_
            data[k] = np.array(data[k], dtype=dtype)

    length = len(data["reward"])
    assert all([len(data[k]) == length for k in data.keys()])

    with io.BytesIO() as bs:
        np.savez_compressed(bs, **data)
        bs.seek(0)
        with open(path, "wb") as f:
            f.write(bs.read())


def load_data(path: str) -> Dict:
    with open(path, "rb") as f:
        data = np.load(f)
        data = {k: data[k] for k in data.keys()}
    # Unnest the data to get everything in the correct format
    data = utils.nest_dict(data)
    kwargs = data.get("kwargs", dict())
    return data["obs"], data["action"], data["reward"], data["done"], data["feature"], kwargs


def add_to_ep(d: Dict, key: str, value: Any, extend: bool = False) -> None:
    # I don't really like this function because it modifies d in place.
    # Perhaps later this can be refactored.
    # If this key isn't the dict, we need to append it
    if key not in d:
        if isinstance(value, dict):
            d[key] = dict()
        else:
            d[key] = list()
    # If the value is a dict, then we need to traverse to the next level.
    if isinstance(value, dict):
        for k, v in value.items():
            add_to_ep(d[key], k, v, extend=extend)
    else:
        if extend:
            d[key].extend(value)
        else:
            d[key].append(value)


def add_dummy_transition(d: Dict, length: int):
    # Helper method to add the dummy transition if it wasn't already
    for k in d.keys():
        if isinstance(d[k], dict):
            add_dummy_transition(d[k], length)
        elif isinstance(d[k], list):
            assert len(d[k]) == length or len(d[k]) == length - 1
            if len(d[k]) == length - 1:
                d[k].insert(0, d[k][0])  # Duplicate the first item.
        else:
            raise ValueError("Invalid value passed to `pad_ep`")


def get_buffer_bytes(buffer: np.ndarray) -> int:
    if isinstance(buffer, dict):
        return sum([get_buffer_bytes(v) for v in buffer.values()])
    elif isinstance(buffer, np.ndarray):
        return buffer.nbytes
    else:
        raise ValueError("Unsupported type passed to `get_buffer_bytes`.")


def remove_stack_dim(space: gym.Space) -> gym.Space:
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict({k: remove_stack_dim(v) for k, v in space.items()})
    elif isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(low=space.low[0], high=space.high[0])
    else:
        return space


class ReplayBuffer(torch.utils.data.IterableDataset):
    """
    Generic Replay Buffer Class.

    This class adheres to the following conventions to support a wide array of multiprocessing options:
    1. Variables/functions starting with "_", like "_help" are to be used by the worker processes. This means
        they should be used only after __iter__ is called.
    2. variables/functions named regularly without a leading "_" are to be used by the main thread. This includes
        standard functions like "add".

    There are a few critical setup options.
    1. Distributed: this determines if the data is stored on the main processes, and then used via the shared address
        space. This will only work when multiprocessing is set to `fork` and not `spawn`.
        AKA it will duplicate memory on Windows and OSX!!!
    2. Capacity: determines if the buffer is setup upon creation. If it is set to a known value, then we can add data
        online with `add`, or by pulling more data from disk. If is set to None, the dataset is initialized to the full
        size of the offline dataset.
    3. batch_size: determines if we use a single sample or return entire batches

    Some options are mutually exclusive. For example, it is bad to use a non-distributed layout with
    workers and online data. This will generate a bunch of copy on writes.

    Data is expected to be stored in a "next" format. This means that data is stored like this:
    s_0, dummy, dummy, dummy
    s_1, a_0  , r_0  , d_0
    s_2, a_1  , r_1  , d_1
    s_3, a_2  , r_2  , d_2 ... End of episode!
    s_0, dummy, dummy, dummy
    s_1, a_0  , r_0  , d_0
    s_2, a_1  , r_1  , d_1

    This format is expected from the load(path) funciton.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_features: int = 16,
        capacity: Optional[int] = None,
        distributed: bool = True,  # Whether or not the dataset is created in __init__ or __iter__. True means _-iter__
        path: Optional[str] = None,
        nstep: int = 1,
        cleanup: bool = True,
        fetch_every: int = 1000,  # How often to pull new data into the replay buffer.
        batch_size: Optional[int] = None,
        sample_multiplier: float = 1.5,  # Should be high enough so we always hit batch_size.
        stack: int = 1,
        pad: int = 0,
        next_obs: bool = True,  # Whether or not to load the next obs.
        stacked_obs: bool = False,  # Whether or not the data provided to the buffer will have stacked obs
        stacked_action: bool = False,  # Whether or not the data provided to the buffer will have stacked obs
    ):
        super().__init__()
        # Check that we don't over add in case of observation stacking
        self.stacked_obs = stacked_obs
        self.stacked_action = stacked_action
        if self.stacked_obs:
            observation_space = remove_stack_dim(observation_space)
        if self.stacked_action:
            action_space = remove_stack_dim(action_space)

        self.observation_space = observation_space
        self.action_space = action_space
        self.num_features = num_features
        self.zero_feature = np.zeros((num_features,), dtype=np.float32)
        self.dummy_action = self.action_space.sample()

        # Data Storage parameters
        self.capacity = capacity  # The total storage of the dataset, or None if growth is disabled
        if self.capacity is not None:
            # Setup a storage path
            self.storage_path = tempfile.mkdtemp(prefix="replay_buffer_")
            print("[research] Replay Buffer Storage Path", self.storage_path)
        self.distributed = distributed

        # Data Fetching parameters
        self.cleanup = cleanup
        self.path = path
        self.fetch_every = fetch_every
        self.sample_multiplier = sample_multiplier
        self.num_episodes = 0

        # Sampling values.
        self.nstep = nstep
        self.stack = stack
        self.batch_size = 1 if batch_size is None else batch_size
        if pad > 0:
            assert self.stack > 1, "Pad > 0 doesn't make sense if we are not padding."
        self.pad = pad
        self.next_obs = next_obs

        if self.capacity is not None:
            # Print the total estimated data footprint used by the replay buffer.
            storage = 0
            storage += utils.np_bytes_per_instance(self.observation_space)
            storage += utils.np_bytes_per_instance(self.action_space)
            storage += utils.np_bytes_per_instance(self.zero_feature)
            storage += utils.np_bytes_per_instance(0.0)  # Reward
            storage += utils.np_bytes_per_instance(False)  # Done
            storage = storage * capacity  # Total storage in Bytes.
            print("[ReplayBuffer] Estimated storage requirement for obs, action, reward, done, feature.")
            print("\t will not include kwarg storage: {:.2f} GB".format(storage / 1024**3))

        # Initialize in __init__ if the replay buffer is not distributed.
        if not self.distributed:
            print("[research] Replay Buffer not distributed. Alloc-ing in __init__")
            self._alloc()

    def _data_generator(self):
        """
        Can be overridden in order to load the initial data differently.
        By default assumes the data to be the standard format, and returned as:
        *(obs, action, reward, done, feature, kwargs)
        or
        None

        This function can be overriden by sub-classes in order to produce data batches.
        It should do the following:
        1. split data across torch data workers
        2. randomize the order of data
        3. yield data of the form (obs, action, reward, done, feature, kwargs)
        """
        if self.path is None:
            return
        # By default get all of the file names that are distributed at the correct index
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id

        ep_filenames = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith(".npz")]
        random.shuffle(ep_filenames)  # Shuffle all the filenames

        if num_workers > 1 and len(ep_filenames) == 1:
            print(
                "[ReplayBuffer] Warning: using multiple workers but single replay file. Reduce memory usage by sharding"
                " data with `save` instead of `save_flat`."
            )
        elif num_workers > 1 and len(ep_filenames) < num_workers:
            print("[ReplayBuffer] Warning: using more workers than dataset files.")

        for ep_filename in ep_filenames:
            ep_idx, _ = [int(x) for x in os.path.splitext(ep_filename)[0].split("_")[-2:]]
            # Spread loaded data across workers if we have multiple workers and files.
            if ep_idx % num_workers != worker_id and len(ep_filenames) > 1:
                continue  # Only yield the files belonging to this worker.
            obs, action, reward, done, feature, kwargs = load_data(ep_filename)
            yield (obs, action, reward, done, feature, kwargs)

    def _alloc(self):
        """
        This function is responsible for allocating all of the data needed.
        It can be called in __init__ or during __iter___.

        It allocates all of the np buffers used to store data internal.
        It also sets the follow variables:
            _idx: internal _idx for the worker thread
            _size: internal _size of each workers dataset
            _current_data_generator: the offline data generator
            _loaded_all_offline_data: set to True if we don't need to load more offline data
        """
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id
        self._current_data_generator = self._data_generator()

        if self.capacity is not None:
            # If capacity was given, then directly alloc the buffers
            self._capacity = self.capacity // num_workers
            self._obs_buffer = utils.np_dataset_alloc(self.observation_space, self._capacity)
            self._action_buffer = utils.np_dataset_alloc(self.action_space, self._capacity)
            self._reward_buffer = utils.np_dataset_alloc(0.0, self._capacity)
            self._done_buffer = utils.np_dataset_alloc(False, self._capacity)
            self._feature_buffer = utils.np_dataset_alloc(self.zero_feature, self._capacity)
            self._kwarg_buffers = dict()
            self._size = 0
            self._idx = 0

            # Next, write in the alloced data lazily using the generator until we are full.
            preloaded_episodes = 0
            try:
                while self._size < self._capacity:
                    obs, action, reward, done, feature, kwargs = next(self._current_data_generator)
                    self._add_to_buffer(obs, action, reward, done, feature, **kwargs)
                    preloaded_episodes += 1
                self._loaded_all_offline_data = False
            except StopIteration:
                self._loaded_all_offline_data = True  # We reached the end of the available dataset.

        else:
            self._capacity = None
            # Get all of the data and concatenate it together
            data = utils.concatenate(*list(self._current_data_generator), dim=0)
            obs, action, reward, done, feature, kwargs = data
            self._obs_buffer = obs
            self._action_buffer = action
            self._reward_buffer = reward
            self._done_buffer = done
            self._feature_buffer = feature
            self._kwarg_buffers = kwargs
            # Set the size to be the shape of the reward buffer
            self._size = self._reward_buffer.shape[0]
            self._idx = self._size
            self._loaded_all_offline_data = True

        # Print the size of the allocation.
        storage = 0
        storage += get_buffer_bytes(self._obs_buffer)
        storage += get_buffer_bytes(self._action_buffer)
        storage += get_buffer_bytes(self._reward_buffer)
        storage += get_buffer_bytes(self._done_buffer)
        storage += get_buffer_bytes(self._feature_buffer)
        storage += get_buffer_bytes(self._kwarg_buffers)
        print("[ReplayBuffer] Worker {:d} allocated {:.2f} GB".format(worker_id, storage / 1024**3))

    def add(
        self,
        obs: Any,
        action: Optional[Union[Dict, np.ndarray]] = None,
        reward: Optional[float] = None,
        done: Optional[bool] = None,
        feature: Optional[float] = None,
        **kwargs,
    ) -> None:
        # Make sure that if we are adding the first transition, it is consistent
        assert self.capacity is not None, "Tried to extend to a static size buffer."
        assert (action is None) == (reward is None) == (done is None) == (feature is None)

        is_list = isinstance(reward, list) or isinstance(reward, np.ndarray)
        # Take only the last value if we are using stacking.
        # This prevents saving a bunch of extra data.
        if not is_list and self.stacked_obs:
            obs = utils.get_from_batch(obs, -1)
        if not is_list and action is not None and self.stacked_action:
            action = utils.get_from_batch(action, -1)

        if action is None:
            assert not isinstance(reward, (np.ndarray, list)), "Tried to add initial transition in batch mode."
            action = copy.deepcopy(self.dummy_action)
            reward = 0.0
            done = False
            feature = copy.deepcopy(self.zero_feature)

        # Now we have multiple cases based on the transition type and parallelism of the dataset
        if not self.distributed:
            # We can add directly to the storage buffers.
            self._add_to_buffer(obs, action, reward, done, feature, **kwargs)
            if self.cleanup:
                # If we are in cleanup mode, we don't keep the old data around. Immediately return
                return

        if not hasattr(self, "current_ep"):
            self.current_ep = dict()

        add_to_ep(self.current_ep, "obs", obs, is_list)
        add_to_ep(self.current_ep, "action", action, is_list)
        add_to_ep(self.current_ep, "reward", reward, is_list)
        add_to_ep(self.current_ep, "done", done, is_list)
        add_to_ep(self.current_ep, "feature", feature, is_list)
        add_to_ep(self.current_ep, "kwargs", kwargs, is_list)

        is_done = done[-1] if is_list else done
        if is_done:
            # Dump the data
            ep_idx = self.num_episodes
            ep_len = len(self.current_ep["reward"])
            # Check to make sure that kwargs are the same length
            add_dummy_transition(self.current_ep, ep_len)
            self.num_episodes += 1
            ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
            ep_filename = f"{ts}_{ep_idx}_{ep_len}.npz"
            save_data(self.current_ep, os.path.join(self.storage_path, ep_filename))
            self.current_ep = dict()

    def _add_to_buffer(self, obs: Any, action: Any, reward: Any, done: Any, feature: Any, **kwargs) -> None:
        # Can add in batches or serially.
        if isinstance(reward, list) or isinstance(reward, np.ndarray):
            num_to_add = len(reward)
            assert num_to_add > 1, "If inputting lists or arrays should have more than one timestep"
        else:
            num_to_add = 1

        if self._idx + num_to_add > self._capacity:
            # Add all we can at first, then add the rest later
            num_b4_wrap = self._capacity - self._idx
            self._add_to_buffer(
                utils.get_from_batch(obs, 0, num_b4_wrap),
                utils.get_from_batch(action, 0, num_b4_wrap),
                reward[:num_b4_wrap],
                done[:num_b4_wrap],
                feature[:num_b4_wrap],
                **utils.get_from_batch(kwargs, 0, num_b4_wrap),
            )
            self._add_to_buffer(
                utils.get_from_batch(obs, num_b4_wrap, num_to_add),
                utils.get_from_batch(action, num_b4_wrap, num_to_add),
                reward[num_b4_wrap:],
                done[num_b4_wrap:],
                feature[num_b4_wrap:],
                **utils.get_from_batch(kwargs, num_b4_wrap, num_to_add),
            )
        else:
            # Just add to the buffer
            start = self._idx
            end = self._idx + num_to_add
            utils.set_in_batch(self._obs_buffer, obs, start, end)
            utils.set_in_batch(self._action_buffer, action, start, end)
            utils.set_in_batch(self._reward_buffer, reward, start, end)
            utils.set_in_batch(self._done_buffer, done, start, end)
            utils.set_in_batch(self._feature_buffer, feature, start, end)

            for k, v in kwargs.items():
                if k not in self._kwarg_buffers:
                    sample_value = utils.get_from_batch(v, 0) if num_to_add > 1 else v
                    self._kwarg_buffers[k] = utils.np_dataset_alloc(sample_value, self._capacity)
                    print("[ReplayBuffer] Allocated", self._kwarg_buffers[k].bytes / 1024**3, "GB")
                utils.set_in_batch(self._kwarg_buffers[k], v, start, end)

            self._idx = (self._idx + num_to_add) % self._capacity
            self._size = min(self._size + num_to_add, self._capacity)

    def save(self, path: str) -> None:
        """
        Save the replay buffer to the specified path. This is literally just copying the files
        from the storage path to the desired path. By default, we will also delete the original files.
        """
        if self.cleanup:
            print("[research] Warning, attempting to save a cleaned up replay buffer. There are likely no files")
        os.makedirs(path, exist_ok=True)
        srcs = os.listdir(self.storage_path)
        for src in srcs:
            shutil.move(os.path.join(self.storage_path, src), os.path.join(path, src))
        print("Successfully saved", len(srcs), "episodes.")

    def save_flat(self, path):
        """
        Save directly from the buffers instead of from the saved data. This saves everything as a flat file.
        """
        assert self._size != 0, "Trying to flat save a buffer with no data."
        data = {
            "obs": utils.get_from_batch(self._obs_buffer, 0, self._size),
            "action": utils.get_from_batch(self._action_buffer, 0, self._size),
            "reward": self._reward_buffer[: self._size],
            "done": self._done_buffer[: self._size],
            "feature": self._feature_buffer[: self._size],
            "kwargs": utils.get_from_batch(self._kwarg_buffers, 0, self._size),
        }
        os.makedirs(path, exist_ok=True)
        ep_len = len(data["reward"])
        ep_idx = 0
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        ep_filename = f"{ts}_{ep_idx}_{ep_len}.npz"
        save_path = os.path.join(path, ep_filename)
        save_data(data, save_path)
        return save_path

    def __del__(self):
        if not self.cleanup:
            return
        if hasattr(self, "storage_path"):
            paths = [os.path.join(self.storage_path, f) for f in os.listdir(self.storage_path)]
            for path in paths:
                try:
                    os.remove(path)
                except:
                    pass
            try:
                os.rmdir(self.storage_path)
            except:
                pass

    def _fetch_online(self) -> None:
        ep_filenames = sorted([os.path.join(self.storage_path, f) for f in os.listdir(self.storage_path)], reverse=True)
        fetched_size = 0
        for ep_filename in ep_filenames:
            ep_idx, ep_len = [int(x) for x in os.path.splitext(ep_filename)[0].split("_")[-2:]]
            if ep_idx % self._num_workers != self._worker_id:
                continue
            if ep_filename in self._episode_filenames:
                break  # We found something we have already loaded
            if fetched_size + ep_len > self._capacity:
                break  # Cannot fetch more than the size of the replay buffer
            # Load the episode from disk
            obs, action, reward, done, feature, kwargs = load_data(ep_filename)
            self._add_to_buffer(obs, action, reward, done, feature, **kwargs)
            self._episode_filenames.add(ep_filename)
            if self.cleanup:
                try:
                    os.remove(ep_filename)
                except OSError:
                    pass

        # Return the fetched size
        return fetched_size

    def _fetch_offline(self) -> None:
        """
        This simple function fetches a new episode from the offline dataset and adds it to the buffer.
        This is done for each worker.
        """
        try:
            data = next(self._current_data_generator)
        except StopIteration:
            self._current_data_generator = self._data_generator()
            data = next(self._current_data_generator)

        obs, action, reward, done, feature, kwargs = data
        self._add_to_buffer(obs, action, reward, done, feature, **kwargs)
        # Return the fetched size
        return len(reward)

    def __iter__(self):
        assert not hasattr(self, "_iterated"), "__iter__ called twice!"
        self._iterated = True
        if self.distributed:
            # Allocate the buffer here if we are distributing across workers.
            self._alloc()

        # Setup variables for _fetch methods for getting new online data
        worker_info = torch.utils.data.get_worker_info()
        self._num_workers = worker_info.num_workers if worker_info is not None else 1
        self._worker_id = worker_info.id if worker_info is not None else 0
        assert self.distributed == (worker_info is not None), "ReplayBuffer.distributed set incorrectly."

        self._episode_filenames = set()
        self._samples_since_last_load = 0
        self._learning_online = False
        while True:
            yield self.sample(batch_size=self.batch_size, stack=self.stack, pad=self.pad)
            # Fetch new data...
            if self._capacity is not None:
                self._samples_since_last_load += 1
                if self._samples_since_last_load >= self.fetch_every:
                    # Fetch offline data
                    if not self._loaded_all_offline_data and not self._learning_online:
                        self._fetch_offline()
                    if self.distributed:  # If we are distributed we need to fetch the data.
                        fetch_size = self._fetch_online()
                        self._learning_online = fetch_size > 0
                    # Reset the fetch counter for this worker.
                    self._samples_since_last_load = 0

    def _get_one_idx(self, stack: int, pad: int) -> Union[int, np.ndarray]:
        # Add 1 for the first dummy transition
        idx = np.random.randint(0, self._size - self.nstep * stack) + 1
        done_idxs = idx + np.arange(self.nstep * (stack - pad)) - 1
        if np.any(self._done_buffer[done_idxs]):
            # If the episode is done at any point in the range, we need to sample again!
            # Note that we removed the pad length, as we can check the padding later
            return self._get_one_idx(stack, pad)
        if stack > 1:
            idx = idx + np.arange(stack) * self.nstep
        return idx

    def _get_many_idxs(self, batch_size: int, stack: int, pad: int, depth: int = 0) -> np.ndarray:
        idxs = np.random.randint(0, self._size - self.nstep * stack, size=int(self.sample_multiplier * batch_size)) + 1

        done_idxs = np.expand_dims(idxs, axis=-1) + np.arange(self.nstep * (stack - pad)) - 1
        valid = np.logical_not(
            np.any(self._done_buffer[done_idxs], axis=-1)
        )  # Compute along the done axis, not the index axis.

        valid_idxs = idxs[valid == True]  # grab only the idxs that are still valid.
        if len(valid_idxs) < batch_size and depth < 20:  # try a max of 20 times
            print(
                "[research ReplayBuffer] Buffer Sampler did not recieve batch_size number of valid indices. Consider"
                " increasing sample_multiplier."
            )
            return self._get_many_idxs(batch_size, stack, pad, depth=depth + 1)
        idxs = valid_idxs[:batch_size]  # Return the first [:batch_size] of them.
        if stack > 1:
            stack_idxs = np.arange(stack) * self.nstep
            idxs = np.expand_dims(idxs, axis=-1) + stack_idxs
        return idxs

    def _compute_mask(self, idxs: np.ndarray) -> np.ndarray:
        # Check the validity via the done buffer to determine the padding mask
        mask = np.zeros(idxs.shape, dtype=np.bool_)
        for i in range(self.nstep):
            mask = (
                mask + self._done_buffer[idxs + (i - 1)]
            )  # Subtract one when checking for parity with index sampling.
        # Now set everything past the first true to be true
        mask = np.minimum(np.cumsum(mask, axis=-1), 1.0)
        return mask

    def sample(self, batch_size: Optional[int] = None, stack: int = 1, pad: int = 0) -> Dict:
        if self._size <= self.nstep * stack + 2:
            return {}
        # NOTE: one small bug is that we won't end up being able to sample segments that span
        # Across the barrier of the replay buffer. We lose 1 to self.nstep transitions.
        # This is only a problem if we keep the capacity too low.
        if batch_size > 1:
            idxs = self._get_many_idxs(batch_size, stack, pad)
        else:
            idxs = self._get_one_idx(stack, pad)
        obs_idxs = idxs - 1
        next_obs_idxs = idxs + self.nstep - 1

        obs = utils.get_from_batch(self._obs_buffer, obs_idxs)
        action = utils.get_from_batch(self._action_buffer, idxs)
        reward = utils.get_from_batch(self._reward_buffer, idxs)
        feature = utils.get_from_batch(self._feature_buffer, idxs)

        kwargs = utils.get_from_batch(self._kwarg_buffers, next_obs_idxs)
        if self.next_obs:
            kwargs["next_obs"] = utils.get_from_batch(self._obs_buffer, next_obs_idxs)

        batch = dict(obs=obs, action=action, reward=reward, feature=feature, **kwargs)
        if pad > 0:
            batch["mask"] = self._compute_mask(idxs)

        return batch

class PairwiseComparisonDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        path: Optional[str] = None,
        discount: float = 0.99,
        nstep: int = 1,
        segment_size: int = 20,
        subsample_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        capacity: Optional[int] = None,
    ):
        super().__init__()
        self.discount = discount
        self.batch_size = 1 if batch_size is None else batch_size
        self.segment_size = segment_size
        self.subsample_size = subsample_size
        self._capacity = capacity

        if self._capacity is None:
            assert path is not None, "If capacity is not given, must have path to load from"
            with open(path, "rb") as f:
                data = np.load(f)
                data = utils.nest_dict(data)
            print(data["obs_1"].shape, observation_space)
            # Set the buffers to be the stored data. Woot woot.
            self.obs_1_buffer = utils.remove_float64(data["obs_1"])
            self.obs_2_buffer = utils.remove_float64(data["obs_2"])
            self.action_1_buffer = utils.remove_float64(data["action_1"])
            self.action_2_buffer = utils.remove_float64(data["action_2"])
            self.label_buffer = utils.remove_float64(data["label"])
            self._size = len(self.label_buffer)
        else:
            # Construct the buffers
            self.obs_1_buffer = utils.np_dataset_alloc(
                observation_space, self._capacity, begin_pad=(self.segment_size,)
            )
            self.obs_2_buffer = utils.np_dataset_alloc(
                observation_space, self._capacity, begin_pad=(self.segment_size,)
            )
            self.action_1_buffer = utils.np_dataset_alloc(action_space, self._capacity, begin_pad=(self.segment_size,))
            self.action_2_buffer = utils.np_dataset_alloc(action_space, self._capacity, begin_pad=(self.segment_size,))
            self.label_buffer = utils.np_dataset_alloc(0.5, self._capacity)
            self._size = 0
            self._idx = 0
            if path is not None:
                assert path is not None, "If capacity is not given, must have path to load from"
                with open(path, "rb") as f:
                    data = np.load(f)
                    data = {k: data[k] for k in data.keys()}
                    dataset_size = data["label"].shape[0]
                    if dataset_size > self._capacity:
                        # Trim the dataset down
                        data = utils.get_from_batch(data, 0, self._capacity)
                    data = utils.nest_dict(data)
                    self.add(data, data["label"])  # Add to the buffer via the add method!

        # Print the size of the allocation.
        storage = 0
        storage += 2 * get_buffer_bytes(self.obs_1_buffer)
        storage += 2 * get_buffer_bytes(self.action_1_buffer)
        storage += get_buffer_bytes(self.label_buffer)
        print("[PairwiseComparisonDataset] allocated {:.2f} GB".format(storage / 1024**3))

        # Clip everything
        lim = 1 - 1e-5
        self.action_1_buffer = np.clip(self.action_1_buffer, a_min=-lim, a_max=lim)
        self.action_2_buffer = np.clip(self.action_2_buffer, a_min=-lim, a_max=lim)

    def add(self, queries: Dict, labels: np.ndarray):
        assert self._capacity is not None, "Can only add to non-static buffers."
        assert (
            torch.utils.data.get_worker_info() is None
        ), "Cannot add to PairwiseComparisonDataset when parallelism is enabled."
        num_to_add = labels.shape[0]
        if self._idx + num_to_add > self._capacity:
            # We have more segments than capacity allows, complete in two writes.
            num_b4_wrap = self._capacity - self._idx
            self.add(utils.get_from_batch(queries, 0, num_b4_wrap), labels[:num_b4_wrap])
            self.add(utils.get_from_batch(queries, num_b4_wrap, num_to_add), labels[num_b4_wrap:])
        else:
            start, end = self._idx, self._idx + num_to_add
            utils.set_in_batch(self.obs_1_buffer, queries["obs_1"], start, end)
            utils.set_in_batch(self.obs_2_buffer, queries["obs_2"], start, end)
            utils.set_in_batch(self.action_1_buffer, queries["action_1"], start, end)
            utils.set_in_batch(self.action_2_buffer, queries["action_2"], start, end)
            self.label_buffer[start:end] = labels
            self._idx = (self._idx + num_to_add) % self._capacity
            self._size = min(self._size + num_to_add, self._capacity)

    def _sample(self, idxs):
        if self.subsample_size is None:
            obs_1 = utils.get_from_batch(self.obs_1_buffer, idxs)
            obs_2 = utils.get_from_batch(self.obs_2_buffer, idxs)
            action_1 = utils.get_from_batch(self.action_1_buffer, idxs)
            action_2 = utils.get_from_batch(self.action_2_buffer, idxs)
            label = self.label_buffer[idxs]
        else:
            # Note: subsample sequences currently do not support arbitrary obs/action spaces.
            # we could do this with two utils calls, but that would be slower.
            start = np.random.randint(0, self.segment_size - self.subsample_size)
            end = start + self.subsample_size
            obs_1 = self.obs_1_buffer[idxs, start:end]
            obs_2 = self.obs_2_buffer[idxs, start:end]
            action_1 = self.action_1_buffer[idxs, start:end]
            action_2 = self.action_2_buffer[idxs, start:end]
            label = self.label_buffer[idxs]

        batch_size = len(idxs)
        discount = self.discount * np.ones(batch_size, dtype=np.float32) if self.batch_size > 1 else self.discount
        return dict(obs_1=obs_1, obs_2=obs_2, action_1=action_1, action_2=action_2, label=label, discount=discount)

    def save(self, path):
        # Save everything to the path via savez
        data = dict(
            obs_1=utils.get_from_batch(self.obs_1_buffer, 0, self._size),
            obs_2=utils.get_from_batch(self.obs_2_buffer, 0, self._size),
            action_1=utils.get_from_batch(self.action_1_buffer, 0, self._size),
            action_2=utils.get_from_batch(self.action_2_buffer, 0, self._size),
            label=self.label_buffer[: self._size],
        )
        data = utils.flatten_dict(data)
        with io.BytesIO() as bs:
            np.savez_compressed(bs, **data)
            bs.seek(0)
            with open(path, "wb") as f:
                f.write(bs.read())

    def __len__(self):
        return self._size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        chunk_size = len(self) // num_workers
        my_inds = np.arange(chunk_size * worker_id, chunk_size * (worker_id + 1))
        idxs = np.random.permutation(my_inds)

        for i in range(math.ceil(len(idxs) / self.batch_size)):  # Need to use ceil to get all data points.
            if self.batch_size == 1:
                cur_idxs = idxs[i]
            else:
                # Might be some overlap here but its probably OK.
                cur_idxs = idxs[i * self.batch_size : min((i + 1) * self.batch_size, len(self))]
            yield self._sample(cur_idxs)
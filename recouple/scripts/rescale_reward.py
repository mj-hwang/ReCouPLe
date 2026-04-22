import numpy as np
import os

def rescale_reward_dataset(data_path):
    # Check if the input file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The file {data_path} does not exist.")
    
    # Load the original npz file
    data = np.load(data_path)
    
    # Check if the "reward" key exists in the dataset
    if "reward" not in data:
        raise KeyError("The dataset does not contain a 'reward' key.")

    # Print the reward statistics
    rewards = data["reward"]
    print("### Original reward statistics: ###")
    print(f"Min reward: {rewards.min()}")
    print(f"Max reward: {rewards.max()}")
    print(f"Mean reward: {rewards.mean()}")
    print(f"Std reward: {rewards.std()}")
    
    # Rescale the rewards to the range [0, 1]
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    scaled_rewards = (rewards - min_reward) / (max_reward - min_reward)
    
    # Generate the new file path with "_rescaled" suffix
    base, ext = os.path.splitext(data_path)
    new_data_path = f"{base}_rescaled{ext}"
    
    # Save the rescaled rewards to the new file
    data = {key: value for key, value in data.items() if key != "reward"}
    np.savez(new_data_path, **data, reward=scaled_rewards)
    print(f"Rescaled rewards saved to {new_data_path}")

if __name__ == "__main__":
    # data_path = "offline_dataset/data_offlinerl_push-v2_gt_seed_1_50000.npz"
    # data_path = "offline_dataset/data_offlinerl_push-v2_mtpiql_seed_1_200000.npz"
    # data_path = "offline_dataset/data_offlinerl_push-v2_recouple_seed_1_200000.npz"

    paths = [
        "offline_dataset/data_offlinerl_push-v2_mtpiql_seed_2_50000.npz",
        "offline_dataset/data_offlinerl_push-v2_mtpiql_seed_2_100000.npz",
        "offline_dataset/data_offlinerl_push-v2_mtpiql_seed_2_200000.npz",
        "offline_dataset/data_offlinerl_push-v2_mtpiql_seed_3_50000.npz",
        "offline_dataset/data_offlinerl_push-v2_mtpiql_seed_3_100000.npz",
        "offline_dataset/data_offlinerl_push-v2_mtpiql_seed_3_200000.npz",
        "offline_dataset/data_offlinerl_push-v2_recouple_seed_2_50000.npz",
        "offline_dataset/data_offlinerl_push-v2_recouple_seed_2_100000.npz",
        "offline_dataset/data_offlinerl_push-v2_recouple_seed_2_200000.npz",
        "offline_dataset/data_offlinerl_push-v2_recouple_seed_3_50000.npz",
        "offline_dataset/data_offlinerl_push-v2_recouple_seed_3_100000.npz",
        "offline_dataset/data_offlinerl_push-v2_recouple_seed_3_200000.npz",
        "offline_dataset/data_offlinerl_push-v2_rpl_seed_1_50000.npz",
        "offline_dataset/data_offlinerl_push-v2_rpl_seed_1_100000.npz",
        "offline_dataset/data_offlinerl_push-v2_rpl_seed_1_200000.npz",
        "offline_dataset/data_offlinerl_push-v2_rpl_seed_2_50000.npz",
        "offline_dataset/data_offlinerl_push-v2_rpl_seed_2_100000.npz",
        "offline_dataset/data_offlinerl_push-v2_rpl_seed_2_200000.npz",
        "offline_dataset/data_offlinerl_push-v2_rpl_seed_3_50000.npz",
        "offline_dataset/data_offlinerl_push-v2_rpl_seed_3_100000.npz",
        "offline_dataset/data_offlinerl_push-v2_rpl_seed_3_200000.npz",
    ]

    for path in paths:
        print(f"Processing {path}...")
        rescale_reward_dataset(path)
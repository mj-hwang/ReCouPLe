import argparse
import os
import pickle
import random
from typing import Optional

import gym
import metaworld
import numpy as np
from metaworld import Task, policies
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

from research.datasets.ipl_dataset import ReplayBuffer
from research.utils.evaluate import EvalMetricTracker

CROSS_ENV_DICT = {
    "pick-place-v2": "pick-place-wall-v2",
    "pick-place-wall-v2": "pick-place-v2",
    "push-v2": "push-wall-v2",
    "push-wall-v2": "push-v2",
    "reach-v2": "reach-wall-v2",
    "reach-wall-v2" : "reach-v2",
}

REASON_FEATURE_DICT = {
    "pick-place-v2": ["H_GG", "L", "I_goal_L", "S"],
    "pick-place-wall-v2": ["H_GM", "L", "I_mid_L_noPW", "L_PW", "I_goal_L_PW", "S"],
    "push-v2": ["G", "T", "G_T", "I_goal_T", "S"],
    "push-wall-v2": ["G", "T", "I_mid_T", "T_PW", "I_goal_T_PW", "S"],
    # "reach-v2": ["IG"],
    # "reach-wall-v2": ["IG"],
}

def collect_episode(
    env: gym.Env,
    policy_env: gym.Env,
    policy,
    dataset: ReplayBuffer,
    metric_tracker: EvalMetricTracker,
    epsilon: float = 0.0,
    noise_type: str = "gaussian",
    init_obs: Optional[np.ndarray] = None,
    env_name: Optional[str] = None,
):
    if init_obs is None:
        obs = env.reset()
    else:
        obs = init_obs
    policy_obs = policy_env.reset()

    dataset.add(obs)
    episode_length = 0
    success_steps = 25
    done = False

    metric_tracker.reset()

    while not done:
        action = policy.get_action(policy_obs)
        if noise_type == "gaussian":
            action = action + epsilon * np.random.randn(*action.shape)
        elif noise_type == "uniform":
            action = action + epsilon * policy_env.action_space.sample()
        elif noise_type == "random":
            action = policy_env.action_space.sample()
        else:
            raise ValueError("Invalid noise type provided.")

        action = np.clip(action, -1 + 1e-5, 1 - 1e-5)  # Clip the action to the valid range after noise.
        obs, reward, done, info = env.step(action)
        policy_obs, _, _, _ = policy_env.step(action)
        metric_tracker.step(reward, info)

        episode_length += 1

        if info["success"]:
            success_steps -= 1

        if success_steps == 0:
            done = True  # If we have been successful for a while, break.
        
        feature_dict = info["feature_dict"]
        # feature = [feature_dict[k] if k in feature_dict else 0.0 for k in REASON_FEATURE_DICT[env_name]]
        # breakpoint()
        feature = [feature_dict[k] for k in REASON_FEATURE_DICT[env_name]]
        
        dataset.add(obs, action, reward, done, feature)

    # breakpoint()

if __name__ == "__main__":
    # Only execute this code if this script is called
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="A Metaworld task name, like drawer-open-v2")
    parser.add_argument("--expert-ep", type=int, default=10)
    parser.add_argument("--cross-env-ep", type=int, default=10)
    parser.add_argument("--random-ep", type=int, default=10)
    parser.add_argument("--within-env-ep", type=int, default=40)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.1, 0.3, 0.5], help="magnitude of gaussian noise.")
    # parser.add_argument("--cross-env-epsilons", type=float, nargs="+", default=[0.1, 0.5, 1.0], help="magnitude of gaussian noise for cross envs.")
    parser.add_argument("--path", "-p", type=str, required=True, help="output path")

    # parser.add_argument("--env", type=str, required=True, help="A Metaworld task name, like drawer-open-v2")
    # parser.add_argument("--expert-ep", type=int, default=100)
    # parser.add_argument("--cross-env-ep", type=int, default=100)
    # parser.add_argument("--epsilons", type=float, nargs="+", default=[0.1], help="magnitude of gaussian noise.")
    # parser.add_argument("--path", "-p", type=str, required=True, help="output path")

    args = parser.parse_args()

    # Make the path, do not double write datasets
    os.makedirs(args.path, exist_ok=False)

    env = gym.make("mw_" + args.env)

    ep_length = env.unwrapped._max_episode_steps + 2
    total_ep = args.expert_ep * len(args.epsilons) + args.cross_env_ep + args.within_env_ep # + args.random_ep

    dataset = ReplayBuffer(
        env.observation_space, 
        env.action_space,
        num_features=len(REASON_FEATURE_DICT[args.env]),
        capacity=total_ep * ep_length, 
        distributed=False
    )

    # Construct the same environment
    observable_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[args.env + "-goal-observable"]()

    # Create the expert policy
    policy_name = "".join([s.capitalize() for s in args.env.split("-")])
    policy_name = policy_name.replace("PegInsert", "PegInsertion")
    policy_name = "Sawyer" + policy_name + "Policy"
    policy = vars(policies)[policy_name]()

    cross_env_name = CROSS_ENV_DICT[args.env]
    cross_observable_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[cross_env_name + "-goal-observable"]()
    cross_policy_name = "".join([s.capitalize() for s in cross_env_name.split("-")])
    cross_policy_name = cross_policy_name.replace("PegInsert", "PegInsertion")
    cross_policy_name = "Sawyer" + cross_policy_name + "Policy"
    cross_policy = vars(policies)[cross_policy_name]()

    default_eps = args.epsilons[0]

    for eps in args.epsilons:
        print("Epsilon: ", eps)
        metric_tracker_expert = EvalMetricTracker()
        for i in range(args.expert_ep):
            obs = env.reset()

            # Get the the random vector
            _last_rand_vec = env.unwrapped._last_rand_vec
            data = dict(rand_vec=_last_rand_vec)
            data["partially_observable"] = False
            data["env_cls"] = type(env.unwrapped._env)
            task = Task(env_name=args.env, data=pickle.dumps(data))  # POTENTIAL ERROR
            observable_env.set_task(task)

            collect_episode(
                env,
                observable_env,
                policy,
                dataset,
                metric_tracker_expert,
                epsilon=eps,
                noise_type="gaussian",
                init_obs=obs,
                env_name=args.env,
            )
            if (i + 1) % 10 == 0:
                print("Finished", i + 1, "expert ep.")
        metrics = metric_tracker_expert.export()
        print(f"Expert (w/ eps: {eps}) Metrics:")
        print(metrics)
        with open(os.path.join(args.path, f"metrics_expert_eps_{eps}.txt"), "a") as f:
            for k, v in metrics.items():
                f.write(k + ": " + str(v) + "\n")
    
    # Now collect the other episodes
    metric_tracker_within_env = EvalMetricTracker()
    observable_env._freeze_rand_vec = False  # Unfreeze the random vector.
    for i in range(args.within_env_ep):
        collect_episode(
            env,
            observable_env,
            policy,
            dataset,
            metric_tracker_within_env,
            epsilon=default_eps,
            noise_type="gaussian",
            init_obs=None,
            env_name=args.env,
        )
        if (i + 1) % 10 == 0:
            print("Finished", i + 1, "within env ep.")
    metrics = metric_tracker_within_env.export()
    print("Within Env Metrics:")
    print(metrics)
    with open(os.path.join(args.path, "metrics_within_ep.txt"), "a") as f:
        for k, v in metrics.items():
            f.write(k + ": " + str(v) + "\n")

    metric_tracker_random = EvalMetricTracker()
    for i in range(args.random_ep):
        collect_episode(
            env,
            observable_env,
            policy,
            dataset,
            metric_tracker_random,
            epsilon=default_eps,
            noise_type="random",
            init_obs=None,
            env_name=args.env,
        )
        if (i + 1) % 10 == 0:
            print("Finished", i + 1, "random ep.")
    metrics = metric_tracker_random.export()
    print("Random Metrics:")
    print(metrics)
    with open(os.path.join(args.path, "metrics_random.txt"), "a") as f:
        for k, v in metrics.items():
            f.write(k + ": " + str(v) + "\n")

    metric_tracker_cross_env = EvalMetricTracker()
    env_names = [name[: -len("-goal-observable")] for name in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys()]
    for i in range(args.cross_env_ep):
        collect_episode(
            env,
            cross_observable_env,
            cross_policy,
            dataset,
            metric_tracker_cross_env,
            epsilon=default_eps,
            noise_type="gaussian",
            init_obs=None,
            env_name=args.env,
        )
        if (i + 1) % 10 == 0:
            print("Finished", i + 1, "cross env ep.")
    metrics = metric_tracker_cross_env.export()
    print("Cross Env Metrics:")
    print(metrics)
    with open(os.path.join(args.path, "metrics_cross_env.txt"), "a") as f:
        for k, v in metrics.items():
            f.write(k + ": " + str(v) + "\n")

    fname = dataset.save_flat(args.path)
    fname = os.path.basename(fname)

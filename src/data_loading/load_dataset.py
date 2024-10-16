import os
import numpy as np


def load_d4rl_dataset(env_name):
    dir_path = f"dataset/{env_name}"
    dataset_name = "d4rl_dataset.npz"
    dataset = np.load(os.path.join(dir_path, dataset_name))

    return dataset


def load_pair(env_name, pair_name):
    dir_path = f"dataset/{env_name}"
    pair = np.load(os.path.join(dir_path, f"{pair_name}.npz"), allow_pickle=True)

    return pair


"""
return structured array of (s0, s1, mu) pairs
s0, s1 is a structured array of (observations, actions)
mu is a float
"""


def get_processed_data(env_name, pair_name, use_normalized_mu=False):
    dataset = load_d4rl_dataset(env_name)
    observations = dataset["observations"]
    actions = dataset["actions"]

    pair = load_pair(env_name, pair_name)

    processed_data = []

    for entry in pair["data"]:
        s0_idx, s1_idx, mu, normalized_mu = (
            entry["s0"],
            entry["s1"],
            entry["mu"],
            entry["normalized_mu"],
        )

        s0_obs = observations[s0_idx[0] : s0_idx[1]]
        s0_act = actions[s0_idx[0] : s0_idx[1]]
        s1_obs = observations[s1_idx[0] : s1_idx[1]]
        s1_act = actions[s1_idx[0] : s1_idx[1]]
        mu = mu

        if use_normalized_mu:
            mu = normalized_mu

        s0 = np.array(
            list(zip(observations, s0_act)),
            dtype=[
                ("observations", "f4", (s0_obs.shape[1],)),
                ("actions", "f4", (s0_act.shape[1],)),
            ],
        )
        s1 = np.array(
            list(zip(observations, s1_act)),
            dtype=[
                ("observations", "f4", (s1_obs.shape[1],)),
                ("actions", "f4", (s1_act.shape[1],)),
            ],
        )

        processed_data.append(
            (
                s0,
                s1,
                mu,
            )
        )

    return np.array(processed_data, dtype=[("s0", "O"), ("s1", "O"), ("mu", "f4")])

import os
from typing import Literal


def make_dir_from_path(path):
    """
    Create directory from path
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_pair_path(
    env_name, exp_name, pair_type: Literal["train", "val", "test"], pair_algo
):
    """
    Return path of pair file
    """
    path = f"pair/{env_name}/{exp_name}/{pair_type}/{pair_algo}.npz"
    make_dir_from_path(path)
    return path


def get_pair_log_path(
    env_name, exp_name, pair_type: Literal["train", "val", "test"], pair_algo, log_file
):
    """
    Return path of pair log file
    """

    path = f"log/{env_name}/{exp_name}/{pair_type}/{pair_algo}/{log_file}"
    make_dir_from_path(path)
    return path


def get_score_model_path(
    env_name,
    exp_name,
    pair_algo,
    score_model: Literal["rnn"],
):
    """
    Return path of score model file
    """
    path = f"model/{env_name}/{exp_name}/score/{score_model}-{pair_algo}.pth"
    make_dir_from_path(path)
    return path


def get_score_model_log_path(
    env_name,
    exp_name,
    pair_algo,
    score_model: Literal["rnn"],
    log_file,
):
    """
    Return path of score model log file
    """
    path = f"log/{env_name}/{exp_name}/score/{score_model}-{pair_algo}/{log_file}"
    make_dir_from_path(path)
    return path


def get_reward_model_path(
    env_name,
    exp_name,
    pair_algo,
    reward_model_algo: Literal["MR", "MR-linear"],
    reward_model_tag,
):
    """
    Return path of reward model file
    """
    path = f"model/{env_name}/{exp_name}/reward/{pair_algo}/{reward_model_algo}-{reward_model_tag}.pth"
    make_dir_from_path(path)
    return path


def get_reward_model_log_path(
    env_name,
    exp_name,
    pair_algo,
    reward_model_algo: Literal["MR", "MR-linear"],
    log_file,
):
    """
    Return path of reward model log file
    """
    path = (
        f"log/{env_name}/{exp_name}/reward/{pair_algo}/{reward_model_algo}/{log_file}"
    )
    make_dir_from_path(path)
    return path


def get_new_dataset_path(env_name, exp_name, pair_algo, reward_model_algo):
    """
    Return path of new dataset file
    """
    path = f"dataset/{env_name}/{exp_name}/{pair_algo}/{reward_model_algo}.npz"
    make_dir_from_path(path)
    return path


def get_new_dataset_log_path(
    env_name, exp_name, pair_algo, reward_model_algo, log_file
):
    """
    Return path of new dataset log file
    """
    path = (
        f"log/{env_name}/{exp_name}/dataset/{pair_algo}/{reward_model_algo}/{log_file}"
    )
    make_dir_from_path(path)
    return path


def get_policy_model_path(
    env_name,
    exp_name,
    pair_algo,
    reward_model_algo: Literal["MR", "MR-linear"],
):
    """
    Return path of policy model file
    """
    path = f"model/{env_name}/{exp_name}/policy/{pair_algo}/{reward_model_algo}/"
    make_dir_from_path(path)
    return path


def get_policy_model_log_path(
    env_name,
    exp_name,
    pair_algo,
    reward_model_algo: Literal["MR", "MR-linear"],
    log_file,
):
    """
    Return path of policy model log file
    """
    path = (
        f"log/{env_name}/{exp_name}/policy/{pair_algo}/{reward_model_algo}/{log_file}"
    )
    make_dir_from_path(path)
    return path
import os

from matplotlib import pyplot as plt
import pandas as pd
from utils import get_policy_model_path, get_policy_model_log_path


def remove_max_min(series):
    # if len(series) <= 10:
    #     return series
    # return series.sort_values().iloc[5:-5]
    return series


def process_csv_files(csv_files):
    combined_df = pd.DataFrame()

    for file in csv_files:
        df = pd.read_csv(file)

        selected_columns = ["Timesteps", "Reward", "Success"]
        if all(col in df.columns for col in selected_columns):
            df = df[selected_columns]
            combined_df = pd.concat([combined_df, df], axis=0)
        else:
            print(f"File {file} does not contain required columns: {selected_columns}")

    grouped = combined_df.groupby("Timesteps")
    mean_df = (
        grouped["Reward"]
        .apply(remove_max_min)
        .groupby("Timesteps")
        .mean()
        .reset_index()
        .rename(columns={"Reward": "Reward_mean"})
    )
    std_df = (
        grouped["Reward"]
        .apply(remove_max_min)
        .groupby("Timesteps")
        .apply(lambda x: (((x**2).mean() - x.mean() ** 2) ** 0.5) / ((len(x)) ** 0.5))
        .reset_index()
        .rename(columns={"Reward": "Reward_std"})
    )
    success_mean_df = (
        grouped["Success"]
        .apply(remove_max_min)
        .groupby("Timesteps")
        .apply(lambda x: (x > 0).mean())
        .reset_index()
        .rename(columns={"Success": "Success_mean"})
    )
    success_std_df = (
        grouped["Success"]
        .apply(remove_max_min)
        .groupby("Timesteps")
        .apply(lambda x: (x > 0))
        .groupby("Timesteps")
        .apply(lambda x: (((x**2).mean() - x.mean() ** 2) ** 0.5) / ((len(x)) ** 0.5))
        .reset_index()
        .rename(columns={"Success": "Success_std"})
    )

    final_df = (
        mean_df.merge(std_df, on="Timesteps")
        .merge(success_mean_df, on="Timesteps")
        .merge(success_std_df, on="Timesteps")
    )

    final_df["Reward_mean"] = final_df["Reward_mean"].ewm(alpha=0.2).mean()
    final_df["Reward_std"] = final_df["Reward_std"].ewm(alpha=0.2).mean()
    final_df["Success_mean"] = final_df["Success_mean"].ewm(alpha=0.2).mean()
    final_df["Success_std"] = final_df["Success_std"].ewm(alpha=0.2).mean()

    return final_df


def plot_and_save(df_list, output_path):
    plt.figure(figsize=(10, 12))
    output_name = output_path.split(".png")[0]

    # Reward
    plt.subplot(2, 1, 1)
    for mu_algo_name, df in df_list:
        mean_values = df["Reward_mean"].values
        std_values = df["Reward_std"].values

        plt.plot(df["Timesteps"].values, mean_values, label=mu_algo_name)
        plt.fill_between(
            df["Timesteps"],
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.2,
        )
    plt.xlabel("Timestep")
    plt.ylabel("Reward per timesteps")
    plt.title(f"{output_name} - Reward")
    plt.legend()
    plt.grid(True)

    # Success
    plt.subplot(2, 1, 2)
    for mu_algo_name, df in df_list:
        mean_values = df["Success_mean"].values
        std_values = df["Success_std"].values

        plt.plot(df["Timesteps"].values, mean_values, label=mu_algo_name)
        plt.fill_between(
            df["Timesteps"],
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.2,
        )
    plt.xlabel("Timestep")
    plt.ylabel("Success rate")
    plt.title(f"{output_name} - Success")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot(env_name, exp_list, postfix_list, output_path):
    csv_files = []
    df_list = []

    for postfix in postfix_list:
        csv_files = []
        for exp_name in exp_list:
            pair_algo = postfix.split("_")[0]
            reward_model_algo = postfix.split("_")[1]
            file_path = get_policy_model_path(
                env_name=env_name,
                exp_name=exp_name,
                pair_algo=pair_algo,
                reward_model_algo=reward_model_algo,
            )
            file_path = f"{file_path}train_log.csv"
            if os.path.exists(file_path):
                print(file_path)
                csv_files.append(file_path)

        if not csv_files:
            continue

        df = process_csv_files(csv_files=csv_files)
        df_list.append((postfix, df))

    if df_list:
        plot_and_save(df_list=df_list, output_path=output_path)
    else:
        print("nothing to plot")


def plot_policy_models(exp_name):
    """
    plot the policy models from hard-coded lists
    """
    env_list = ["box-close-v2", "button-press-topdown-wall-v2", "sweep-into-v2", "drawer-open-v2"]
    exp_list = [
        f"{exp_name}-00",
        f"{exp_name}-01",
        f"{exp_name}-02",
        f"{exp_name}-03",
        f"{exp_name}-04",
    ]
    postfix_list = [
        "full-binary_MR-linear",
        "full-binary-with-0.5_MR-linear",
        "lstm.exp-full-binary_MR-linear",
        "lstm.exp-aug-10000-full-binary_MR-linear",
        "lstm.exp-aug-50000-full-binary_MR-linear",
    ]

    for env_name in env_list:
        plot(
            env_name=env_name,
            exp_list=exp_list,
            postfix_list=postfix_list,
            output_path=get_policy_model_log_path(
                env_name=env_name,
                exp_name=exp_name,
                pair_algo="all",
                reward_model_algo="all",
                log_file="train_log.png",
            ),
        )

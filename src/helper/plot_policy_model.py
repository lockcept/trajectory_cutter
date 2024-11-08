import os

from matplotlib import pyplot as plt
import pandas as pd


def process_csv_files(csv_files, label):
    combined_df = pd.DataFrame()

    for file in csv_files:
        df = pd.read_csv(file)

        selected_columns = ["timestep", label, f"{label}_std"]
        if all(col in df.columns for col in selected_columns):
            df = df[selected_columns]
            combined_df = pd.concat([combined_df, df], axis=0)
        else:
            print(f"File {file} does not contain required columns: {selected_columns}")

    grouped = combined_df.groupby("timestep")
    mean_df = grouped[label].mean().reset_index()
    std_df = grouped[f"{label}_std"].apply(lambda x: (x**2).mean() ** 0.5).reset_index()

    result = pd.merge(mean_df, std_df, on="timestep")
    smoothing_factor = 0.3  # 스무딩 팩터
    result["smoothed_mean"] = result[label].ewm(alpha=smoothing_factor).mean()
    result["smoothed_std"] = result[f"{label}_std"].ewm(alpha=smoothing_factor).mean()

    return result


def plot_and_save(df_list=[], output_name="name"):
    plt.figure(figsize=(10, 6))
    for mu_algo_name, df in df_list:
        mean_values = df["smoothed_mean"].values
        std_values = df["smoothed_std"].values

        plt.plot(
            df["timestep"].values, mean_values, label=mu_algo_name
        )  # Plotting the mean column
        plt.fill_between(
            df["timestep"],
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.2,
        )
    label = df_list[0][1].columns[1]

    plt.xlabel("Timestep")
    plt.ylabel("Mean and Standard Deviation")
    plt.title(f"{output_name} - {label}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"log/{output_name}.png")
    plt.close()


def plot(env_name="", pair_list=[], mu_algo_list=[], output_name="name"):
    label = "eval/normalized_episode_reward"

    csv_files = []
    df_list = []

    for mu_algo in mu_algo_list:
        csv_files = []
        for pair_name in pair_list:
            file_path = f"model/{env_name}/policy/{pair_name}_{mu_algo}_MR/record/policy_training_progress.csv"
            if os.path.exists(file_path):
                csv_files.append(file_path)

        df = process_csv_files(
            csv_files=csv_files,
            label=label,
        )
        df_list.append((mu_algo, df))

    plot_and_save(df_list=df_list, output_name=output_name)
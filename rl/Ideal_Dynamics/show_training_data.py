import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_style("whitegrid")


def plot(varname, data_list, labels):
    """Plot a training metric across multiple experiments.

    Args:
        varname:  Column name in the progress CSV to plot.
        data_list: List of DataFrames (one per experiment).
        labels:   List of legend labels (same length as data_list).
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for i, (data, label) in enumerate(zip(data_list, labels)):
        sns.lineplot(data=data[varname].dropna(), ax=ax,
                     label=label, color=f"C{i + 1}")
    plt.xlabel("Training Iterations")


if __name__ == "__main__":
    experiments = [
        ("./CTBR/",    "CTBR"),
        ("./CTBR+DR/", "CTBR+DR"),
        ("./CTBR+ID/", "CTBR+ID"),
    ]

    data_list = [pd.read_csv(f"{d}/progress.csv") for d, _ in experiments]
    labels = [name for _, name in experiments]

    plot(varname="ep_reward_mean", data_list=data_list, labels=labels)

    plt.show()

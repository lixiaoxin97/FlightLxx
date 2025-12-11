import seaborn as sns
import pandas as pd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
#
sns.set_style("whitegrid")
#


def plot(varname):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    #
    sns.lineplot(data=data_1[varname].dropna(), ax=ax, label="BL", color="C{0}".format(1))
    sns.lineplot(data=data_2[varname].dropna(), ax=ax, label="GA", color="C{0}".format(2))
    sns.lineplot(data=data_3[varname].dropna(), ax=ax, label="DR", color="C{0}".format(3))
    sns.lineplot(data=data_4[varname].dropna(), ax=ax, label="OK", color="C{0}".format(4))
    plt.xlabel("Training Iterations")


if __name__ == "__main__":
    
    logger_dir_1 = "./BL/"
    csv_path_1 = logger_dir_1 + "progress.csv"
    data_1 = pd.read_csv(csv_path_1)
    
    logger_dir_2 = "./DR/"
    csv_path_2 = logger_dir_2 + "progress.csv"
    data_2 = pd.read_csv(csv_path_2)

    logger_dir_3 = "./GA/"
    csv_path_3 = logger_dir_3 + "progress.csv"
    data_3 = pd.read_csv(csv_path_3)

    logger_dir_4 = "./OK/"
    csv_path_4 = logger_dir_4 + "progress.csv"
    data_4 = pd.read_csv(csv_path_4)

    # var_name1 = 'ep_len_mean'

    # plot(varname=var_name1)

    var_name2 = 'ep_reward_mean'

    plot(varname=var_name2)

    plt.show()

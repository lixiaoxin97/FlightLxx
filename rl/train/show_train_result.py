import seaborn as sns
import pandas as pd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
#
sns.set_style("whitegrid")
#


def plot(varname, title):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    #
    sns.lineplot(data=data_2[varname].dropna(), ax=ax, label="OPTC")
    sns.lineplot(data=data_1[varname].dropna(), ax=ax, label="BTC")
    plt.xlabel("Training Iterations")
    plt.title(str(title))


if __name__ == "__main__":
    
    logger_dir_1 = "./BTC_1_NP2G/"
    csv_path_1 = logger_dir_1 + "progress.csv"
    data_1 = pd.read_csv(csv_path_1)
    
    logger_dir_2 = "./BTC_2_P2G/"
    csv_path_2 = logger_dir_2 + "progress.csv"
    data_2 = pd.read_csv(csv_path_2)


    var_name = 'ep_len_mean'

    title_name = 'Episode Length'

    plot(varname=var_name, title=title_name)

    plt.show()

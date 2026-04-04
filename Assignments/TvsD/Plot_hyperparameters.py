import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from Helper import LearningCurvePlot, smooth


def mean_and_std(x: np.ndarray):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    return mean, std

def sort_files(directory):

    data_files = {"Policies": ([], []),
                  "Width": ([], []),
                  "Depth": ([], []),
                  "Lr": ([], []),
                  "Data-to-update": ([], []),
                  "Loss": ([], [])}
    
    files_list = os.listdir(directory)
    for file in files_list:

        if file.startswith(("Softmax", "EpsGreedy")):
            if "timesteps" in file:
                data_files["Policies"][0].append(os.path.join(directory, file))
            else:
                data_files["Policies"][1].append(os.path.join(directory, file))

        if file.startswith("Width"):
            if "timesteps" in file:
                data_files["Width"][0].append(os.path.join(directory, file))
            else:
                data_files["Width"][1].append(os.path.join(directory, file))

        if file.startswith("Layers"):
            if "timesteps" in file:
                data_files["Depth"][0].append(os.path.join(directory, file))
            else:
                data_files["Depth"][1].append(os.path.join(directory, file))

        if file.startswith("LR"):
            if "timesteps" in file:
                data_files["Lr"][0].append(os.path.join(directory, file))
            else:
                data_files["Lr"][1].append(os.path.join(directory, file))

        if file.startswith("Batch"):
            if "timesteps" in file:
                data_files["Data-to-update"][0].append(os.path.join(directory, file))
            else:
                data_files["Data-to-update"][1].append(os.path.join(directory, file))

        if file.startswith("loss"):
            if "timesteps" in file:
                data_files["Loss timesteps"][0].append(os.path.join(directory, file))
            else:
                data_files["Loss returns"][1].append(os.path.join(directory, file))

    return data_files
    
def plot_data(data_dict):
    ncols = 3
    nrows = int(np.ceil(len(data_dict) / ncols))

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4 * nrows), sharey=True, sharex=True)
    ax = ax.flatten()
    

    for i, data_files in enumerate(data_dict):
        timesteps, returns = data_dict[data_files]
        ax[i].grid(alpha=0.3)
        ax[i].set_title(data_files)
        if i // ncols > 0:
            ax[i].set_xlabel("Steps")
        if i % ncols == 0:
            ax[i].set_ylabel("Return")
        else:
            ax[i].set_yticks([])
        for t, r in zip(timesteps, returns):
            t = np.load(t)[0]
            r = np.load(r)
            mean, std = mean_and_std(r)
            smooth_mean = np.array(smooth(mean, window=9, poly=2))
            smooth_std = np.array(smooth(std, window=9, poly=2))

            smooth_upper = smooth_mean + smooth_std
            smooth_lower = smooth_mean - smooth_std 
            ax[i].plot(t, smooth_mean, linestyle='--')
            ax[i].fill_between(t, smooth_lower, smooth_upper, alpha=0.2)


    
    ax[0].set_ylabel("Return")

    plt.show()
    

if __name__ == "__main__":
    cwd = os.getcwd()
    directory = os.path.join('Assignments', 'TvsD', 'results', 'combined')

    data_files = sort_files(directory=directory)
    for key in data_files:
        print(f"{key}: {int(len(data_files[key])/2)} configs \n")
    plot_data(data_dict=data_files)

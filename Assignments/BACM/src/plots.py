from typing import Any

from matplotlib.figure import Figure
from utils import smooth
from collections.abc import Mapping
from matplotlib.axes import Axes
import matplotlib.pyplot as plt


def plot_curve(ax: Axes, x, y, window, poly, label="Return", color="blue") -> Axes:
    """Plot a specific curve for a run, allows for inclusion of non-smoothed curve"""
    ax.plot(x, y, alpha=0.3, color="red", label="raw")
    try:
        smoothed = smooth(y, window, poly)
    except Exception as e:
        print("Savgol failed:", e)
        smoothed = y
    ax.plot(x, smoothed, color=color, label=f"smoothed (w={window},p={poly})")
    ax.set_ylabel(label)
    ax.set_xlabel("Episode")
    return ax


def plot_training(training_info: Mapping[str, Any], window, poly) -> Figure:
    """Plot training information, given in a dictionary, which is mostly outputted by the code"""
    plot_names = training_info.keys()
    n_plots = len(plot_names)
    fig, axes = plt.subplots(nrows=n_plots, ncols=1)

    axes = axes.flatten()
    for ax, (name, data) in zip(axes, training_info.items()):
        steps = [data_point[0] for data_point in data]
        y = [data_point[1] for data_point in data]
        plot_curve(ax, steps, y, window=window, poly=poly, label=name)
    plt.legend()
    plt.show()
    return fig

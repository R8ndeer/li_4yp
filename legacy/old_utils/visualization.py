import time
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay


class CustomMetricPlotter:
    def __init__(self, update_interval: int | float, subplot_size, xlabel="Epoch"):
        self.xlabel = xlabel
        self.update_interval = update_interval
        self.tic = time.time()
        self.fig, self.axes = plt.subplots(
            subplot_size[0],
            subplot_size[1],
            figsize=(subplot_size[0] * 5, subplot_size[1] * 5),
        )
        plt.tight_layout(pad=3.0)
        plt.close(self.fig)  # Prevent initial empty plot display

    def __plot_all(self, history: dict, clear_output=True) -> None:
        epochs = range(1, len(history["train_loss"]) + 1)
        metrics = list(history.keys())

        # Handle both single axes and multiple axes
        if hasattr(self.axes, "flat"):
            axes_list = self.axes.flat
        else:
            axes_list = [self.axes] if not hasattr(self.axes, "__iter__") else self.axes

        for i, metric in enumerate(metrics):
            if i < len(axes_list):
                ax = axes_list[i]
                ax.cla()
                ax.plot(epochs, history[metric], label=metric, color="C" + str(i))
                ax.set_title(metric.replace("_", " ").title())
                ax.set_xlabel(self.xlabel)
                ax.legend()
                ax.grid(True)
        if clear_output:
            ipythondisplay.clear_output(wait=True)
        ipythondisplay.display(self.fig)

    def plot(self, history: dict, immediate: bool = False) -> None:
        if immediate:
            self.__plot_all(history)
        if time.time() - self.tic > self.update_interval:
            self.__plot_all(history)
            self.tic = time.time()

    def plot_final(self, history: dict):
        """Plot final results without clearing output to avoid empty figures"""
        self.__plot_all(history, clear_output=False)

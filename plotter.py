import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class Plotter():
    def plot(self, data, title="title", y_axis_title="y axis title", x_axis_title="x axis title"):
        t = np.arange(len(data))
        s = data

        fig, ax = plt.subplots()
        ax.plot(t, s)

        ax.set(xlabel=x_axis_title, ylabel=y_axis_title,
               title=title)
        ax.grid()

        # fig.savefig("test.png")
        plt.show()

    def plot_multiple_signals(self, data_1, data_2, title="title", y_axis_title="y axis title", x_axis_title="x axis title",):
        t = np.arange(len(data_1))
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(t, data_1, t, data_2)
        axs[0].set_xlabel(x_axis_title)
        axs[0].set_ylabel(y_axis_title)
        axs[0].grid(True)

        cxy, f = axs[1].cohere(data_1.flatten(), data_2.flatten(), NFFT=32, Fs=1)
        axs[1].set_ylabel('coherence')
        fig.tight_layout()
        plt.show()

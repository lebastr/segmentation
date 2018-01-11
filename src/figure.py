from typing import Dict

from matplotlib import pyplot as plt
import numpy as np


class Accumulator:
    """ Accumulates history of scalar changes """
    def __init__(self, with_std=False):
        self.history = []
        self.history_std = [] if with_std else None
        self.last = []

    def append(self, y):
        self.last.append(y)

    def accumulate(self):
        self.history.append(np.mean(self.last))
        if self.history_std is not None:
            self.history_std.append(np.std(self.last))
        self.last.clear()

    def accumulate_raw(self, mean, std=None):
        self.history.append(mean)
        if self.history_std is not None:
            self.history_std.append(std)


class Figure:
    """ Window with learning stats"""

    def __init__(self, graph_names=None, accums: Dict[str, Accumulator]=None, title=None):
        if not plt.isinteractive():
            plt.ion()
        if graph_names is None:
            graph_names = []
        if accums is None:
            accums = {}

        self.fig = plt.figure()
        self.graphs = {}
        self.accums = accums

        gi = 1
        for gn in graph_names + list(accums.keys()):
            ax = self.fig.add_subplot(len(graph_names)+len(accums), 1, gi)
            ax.set_ylabel(gn)
            pl = None
            fl = None
            self.graphs[gn] = [ax, pl, fl]  # Axe, Plot, Filling
            gi += 1

        if title is not None:
            self.fig.canvas.set_window_title(title)
        plt.show()

    def plot(self, graph_name, data, fill_data=None):
        """ Update single graph"""
        if isinstance(data, list):
            data = np.array(data)
        if isinstance(fill_data, list):
            fill_data = np.array(fill_data)

        graph = self.graphs[graph_name]
        assert graph
        if graph[1] is None:
            graph[1] = graph[0].plot(data)[0]
        else:
            graph[1].set_xdata(np.arange(0, len(data)))
            graph[1].set_ydata(data)
        if fill_data is not None:
            if graph[2] is not None:
                graph[2].remove()
            graph[2] = graph[0].fill_between(np.arange(0, len(fill_data)),
                                             data - fill_data, data + fill_data, color='b', alpha=0.2)

        graph[0].relim()
        graph[0].autoscale_view()

    def plot_accums(self):
        for gn, acc in self.accums.items():
            self.plot(gn, acc.history, acc.history_std)

    def draw(self):
        """ Redraw after updates"""
        self.fig.canvas.draw()
        plt.pause(0.00001)  # http://stackoverflow.com/a/24228275

    def save(self, path):
        self.fig.savefig(path)

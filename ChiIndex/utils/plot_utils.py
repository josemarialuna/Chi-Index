"""Utilities for result and progress visualization."""

import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, title, xlabel, ylabel, path, filename):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.xdata = []
        self.ydata = []

        self.path = path
        self.filename = filename

    def start(self):
        self._initialize_plot()

    def _initialize_plot(self):
        self.fig = plt.figure(self.title)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel(self.xlabel)  # Establece el título del eje x
        self.ax.set_ylabel(self.ylabel)  # Establece el título del eje y
        self.ax.grid(True)

    def savefig(self, full_path):
        self.fig.savefig(full_path)

    def centroids(self, kmeans, save_image: bool = False):
        cluster_centers_ = kmeans.cluster_centers_
        self._initialize_plot()
        for values in range(len(cluster_centers_)):
            self.ax.plot(range(len(cluster_centers_[0])), cluster_centers_[values], label="Cluster " + str(values))
        self.ax.legend(loc='upper left')

        if save_image:
            path = f'{self.path}/centroids_{self.filename}.png'
            self.savefig(path)

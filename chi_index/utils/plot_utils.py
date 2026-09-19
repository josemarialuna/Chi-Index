"""Optional Matplotlib visualization of cluster centroids."""

from pathlib import Path


class Visualizer:
    def __init__(self, title, xlabel, ylabel, path, filename):
        self.title, self.xlabel, self.ylabel = title, xlabel, ylabel
        self.path, self.filename = Path(path), filename
        self.fig = None

    def start(self):
        self._initialize_plot()

    def _initialize_plot(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError('Plotting requires: pip install "chi-index[plot]"') from exc
        self.close()
        self.fig, self.ax = plt.subplots()
        self.ax.set(title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)
        self.ax.grid(True)

    def savefig(self, full_path):
        if self.fig is None:
            raise RuntimeError("Initialize a plot before saving it.")
        full_path = Path(full_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(full_path, bbox_inches="tight")
        return full_path

    def centroids(self, kmeans, save_image=False):
        self._initialize_plot()
        for index, values in enumerate(kmeans.cluster_centers_):
            self.ax.plot(range(len(values)), values, label=f"Cluster {index}")
        self.ax.legend(loc="upper left")
        if save_image:
            return self.savefig(self.path / f"centroids_{self.filename}.png")
        return None

    def close(self):
        """Release this visualizer's figure without closing other figures."""
        if self.fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self.fig)
            self.fig = None

import numpy as np
import bottleneck as bn
import matplotlib.pyplot as plt
import os

class RTrack:
    def __init__(self, csv_file=None):
        self.reward_hist = []
        self.csv_file = csv_file
        if self.csv_file is not None:
            self._start_csv_file()

    def _start_csv_file(self):
        with open(self.csv_file, "a") as f:
            f.write(f"reward\n")

    def _write_csv(self, reward):
        with open(self.csv_file, "a") as f:
            f.write(f"{reward}\n")
        
    def add(self, reward):
        self.reward_hist.append(reward)
        if self.csv_file is not None:
            self._write_csv(reward)

    def plot(self, window=100, alpha=0.2, save=False, close_plots=False, pre_fix=""):
        plt.figure()
        plt.title(pre_fix+"Mean")
        p = plt.plot(bn.move_mean(self.reward_hist, window=window))[0]
        if alpha > 0:
            plt.plot(self.reward_hist, color=p.get_color(), alpha=alpha)
        plt.xlim(xmin=0)
        plt.grid(True)
        if not save is False:
            plt.savefig(os.path.join(save, "move_mean.svg"))
        if not close_plots:
            plt.show()
        else:
            plt.close()

        plt.figure()
        plt.title(pre_fix+"Min")
        plt.plot(bn.move_min(self.reward_hist, window=window))
        plt.xlim(xmin=0)
        plt.grid(True)
        if not save is False:
            plt.savefig(os.path.join(save, "move_min.svg"))
        if not close_plots:
            plt.show()
        else:
            plt.close()

        plt.figure()
        plt.title(pre_fix+"Max")
        plt.plot(bn.move_max(self.reward_hist, window=window))
        plt.xlim(xmin=0)
        plt.grid(True)
        if not save is False:
            plt.savefig(os.path.join(save, "move_max.svg"))
        if not close_plots:
            plt.show()
        else:
            plt.close()

    def save(self, file_path):
        np.save(file_path, self.reward_hist)


if __name__ == "__main__":
    rt = RTrack()
    for _ in range(1000):
        rt.add(np.random.rand())
    rt.plot(save="")
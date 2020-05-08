import numpy as np
import bottleneck as bn
import matplotlib.pyplot as plt
import os

class RTrack:
    def __init__(self):
        self.reward_hist = []
        
    def add(self, reward):
        self.reward_hist.append(reward)

    def plot(self, window=100, alpha=0.2, save=False, close_plots=False):
        plt.figure()
        plt.title("Mean")
        p = plt.plot(bn.move_mean(self.reward_hist, window=window))[0]
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
        plt.title("Min")
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
        plt.title("Max")
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
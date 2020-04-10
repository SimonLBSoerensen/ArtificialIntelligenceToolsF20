import numpy as np
import bottleneck as bn
import matplotlib.pyplot as plt

class RTrack:
    def __init__(self):
        self.reward_hist = []
        
    def add(self, reward):
        self.reward_hist.append(reward)

    def plot(self):
        plt.figure()
        plt.title("Mean")
        plt.plot(bn.move_mean(self.reward_hist, window=100, min_count=1))
        plt.show()

        plt.figure()
        plt.title("Min")
        plt.plot(bn.move_min(self.reward_hist, window=100, min_count=1))
        plt.show()

        plt.figure()
        plt.title("Max")
        plt.plot(bn.move_max(self.reward_hist, window=100, min_count=1))
        plt.show()
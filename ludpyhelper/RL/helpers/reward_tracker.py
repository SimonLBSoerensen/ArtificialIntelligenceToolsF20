import numpy as np
import bottleneck as bn

class RTrack:
    def __init__(self):
        self.reward_hist = []
        
    def add(self, reward):
        self.reward_hist.append(reward)
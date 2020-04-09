import numpy as np

def triangle_wave(t, min, max, p):
    temp = np.floor(((2*t)/p+max/2))
    return 4/p * (t - p/2*temp) * np.power(min, temp)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    numbers = np.arange(1, 10, 0.1)
    
    triangle_wave_y = [triangle_wave(t, -1, 2, 2) for t in numbers]
    print(triangle_wave_y)
    plt.figure()
    plt.plot(numbers, triangle_wave_y)
    plt.show()
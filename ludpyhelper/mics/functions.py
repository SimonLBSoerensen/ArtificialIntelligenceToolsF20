import numpy as np
from collections.abc import Iterable

def triangle(x, height, start, midt, end):
    if isinstance(x, Iterable):
        return [triangle(el, height, start, midt, end) for el in x]
    else:
        if not start < x < end:
            return 0
        elif x == midt:
            return height
        elif x < midt:
            return height * (x - start) / (midt - start)
        else:
            return height * (end - x) / (end - midt)


def trapezoid(x, height, start, midt_start, midt_end, end):
    if isinstance(x, Iterable):
        return [trapezoid(el, height, start, midt_start, midt_end, end) for el in x]
    else:
        if not start < x < end:
            return 0
        elif x < midt_start:
            return height * min(1, (x - start) / (midt_start - start))
        elif x <= midt_end:
            return height
        elif x < end:
            return height * (end - x) / (end - midt_end)
        else:
            return 0


def rectangle(x, height, start, end):
    if isinstance(x, Iterable):
        return [rectangle(el, height, start, end) for el in x]
    else:
        if start < x < end:
            return height
        else:
            return 0


def discrete(x, height, xmin, ymin, xmax, ymax):
    if isinstance(x, Iterable):
        return [discrete(el, height, xmin, ymin, xmax, ymax) for el in x]
    else:
        return ((height * (ymax - ymin)) / (xmax - xmin)) * (x - xmin) + ymin


def bell(x, height, center, width, slope):
    return height / (1 + np.power((np.abs(x - center) / width), 2 * slope))


def binary(x, height, start, direction):
    if isinstance(x, Iterable):
        return [binary(el, height, start, direction) for el in x]
    else:
        if x >= start and direction > 0:
            return height
        elif x <= start and direction < 0:
            return height
        else:
            return 0


def cosine(x, height, center, width):
    if isinstance(x, Iterable):
        return [cosine(el, height, center, width) for el in x]
    else:
        if x < center - 0.5 * width or x > center + 0.5 * width:
            return 0
        else:
            return 0.5 * height * (1 + np.cos(2.0 / width * np.pi * (x - center)))


def gaussian(x, height, mean, std):
    return height * np.exp(-np.power((x - mean), 2) / (2 * np.power(std, 2)))


def gaussian_product(x, height, mean_a, std_a, mean_b, std_b):
    if isinstance(x, Iterable):
        return [gaussian_product(el, height, mean_a, std_a, mean_b, std_b) for el in x]
    else:
        i = 1 if x <= mean_a else 0
        j = 1 if x >= mean_b else 0
        g1 = (1 - i) + i * np.exp(-np.power((x - mean_a), 2) / (2 * np.power(std_a, 2)))
        g2 = (1 - j) + j * np.exp(-np.power((x - mean_b), 2) / (2 * np.power(std_b, 2)))
        return height * g1 * g2


def pi_shape(x, height, bottom_left, top_left, top_right, bottom_right):
    if isinstance(x, Iterable):
        return [pi_shape(el, height, bottom_left, top_left, top_right, bottom_right) for el in x]
    else:
        if x <= bottom_left:
            return 0
        elif x <= 0.5 * (top_left + bottom_left):
            return 2 * height * np.power((x - bottom_left) / (top_left - bottom_left), 2)
        elif x < top_left:
            return height * (1 - 2 * np.power((x - top_left) / (top_left - bottom_left), 2))
        elif x <= top_right:
            return height
        elif x <= 0.5 * (top_right + bottom_right):
            return height * (1 - 2 * np.power((x - top_right) / (bottom_right - top_right), 2))
        elif x < bottom_right:
            return 2 * height * np.power((x - bottom_right) / (bottom_right - top_right), 2)
        else:
            return 0


def sigmoid_difference(x, height, left_inflection, left_slope, right_inflection, right_slope):
    a = 1 / (1 + np.exp(-left_slope * (x - left_inflection)))
    b = 1 / (1 + np.exp(-right_slope * (x - right_inflection)))
    return height * (a - b)


def sigmoid_product(x, height, left_inflection, left_slope, right_inflection, right_slope):
    a = 1 / (1 + np.exp(-left_slope * (x - left_inflection)))
    b = 1 / (1 + np.exp(-right_slope * (x - right_inflection)))
    return height * (a * b)


def spike(x, height, width, center):
    return height * np.exp(-np.abs(10 / width * (x - center)))


def concave(x, height, inflection, end):
    if isinstance(x, Iterable):
        return [concave(el, height, inflection, end) for el in x]
    else:
        if inflection <= end and x < end:
            return height * (end - inflection) / (2 * end - inflection - x)
        elif inflection > end and x > end:
            return height * (inflection - end) / (-2 * end + inflection + x)
        else:
            return height


def ramp(x, height, start, end):
    if isinstance(x, Iterable):
        return [ramp(el, height, start, end) for el in x]
    else:
        if start == end:
            return 0
        elif start < end:
            if x <= start:
                return 0
            elif x >= end:
                return height
            else:
                return height * (x - start) / (end - start)
        elif start > end:
            if x >= start:
                return 0
            elif x <= end:
                return height
            else:
                return height * (start - x) / (start - end)


def sigmoid(x, height, slope, inflection):
    return height / (1 + np.exp(-slope * (x - inflection)))


def s_shape(x, height, start, end):
    if isinstance(x, Iterable):
        return [s_shape(el, height, start, end) for el in x]
    else:
        if x <= start:
            return 0
        elif x <= 0.5*(start + end):
            return height * (2 * np.power((x - start)/(end - start), 2))
        elif x < end:
            return height * (1 - 2*np.power((x - end)/(end - start), 2))
        else:
            return height


def z_shape(x, height, start, end):
    if isinstance(x, Iterable):
        return [z_shape(el, height, start, end) for el in x]
    else:
        if x <= start:
            return height
        elif x <= 0.5*(start + end):
            return height * (1 - 2*np.power((x - start)/(end - start), 2))
        elif x < end:
            return height * (2 * np.power((x - end)/(end - start), 2))
        else:
            return 0

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xs = np.arange(0, 100)

    plt.figure()
    plt.title("Ramp")
    end = 20
    start = 60
    height = 1.0
    y = ramp(xs, height, start, end)
    plt.plot(xs, y)
    y = ramp(xs, height, end, start)
    plt.plot(xs, y)
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Triangle")
    height = 1
    start = 20
    midt = 60
    end = 80
    y = triangle(xs, height, start, midt, end)
    plt.plot(xs, y)
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Trapezoid")
    height = 1
    start = 20
    midt_start = 60
    midt_end = 70
    end = 90
    y = trapezoid(xs, height, start, midt_start, midt_end, end)
    plt.plot(xs, y)
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Rectangle")
    height = 1
    start = 20
    end = 40
    y = rectangle(xs, height, start, end)
    plt.plot(xs, y)
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Discrete")
    height = 1
    xmin = 20
    xmax = 40
    ymin = 0.1
    ymax = 0.8
    y = discrete(xs, 1, xmin, ymin, xmax, ymax)
    plt.plot(xs, y)
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Bell")
    height = 1
    center = 50
    width = 10
    slope = 5
    y = bell(xs, height, center, width, slope)
    plt.plot(xs, y)
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Binary")
    height = 1
    start = 50
    y = binary(xs, height, start, 1)
    plt.plot(xs, y)
    y = binary(xs, height, start, -1)
    plt.plot(xs, y)
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Cosine")
    height = 1
    center = 50
    width = 100
    y = cosine(xs, height, center, width)
    plt.plot(xs, y)
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Gaussian")
    height = 1
    mean = 50
    std = 10
    y = gaussian(xs, height, mean, std)
    plt.plot(xs, y)
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Gaussian Product")
    height = 1
    mean_a = 50
    std_a = 10
    mean_b = 20
    std_b = 10
    y = gaussian_product(xs, height, mean_a, std_a, mean_b, std_b)
    plt.plot(xs, y)
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Pi Shape")
    height = 1
    bottom_left = 1
    top_left = 50
    top_right = 60
    bottom_right = 80
    y = pi_shape(xs, height, bottom_left, top_left, top_right, bottom_right)
    plt.plot(xs, y)
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Sigmoid Difference")
    height = 1
    left_inflection = 10
    left_slope = 5
    right_inflection = 50
    right_slope = 0.5
    y = sigmoid_difference(xs, height, left_inflection, left_slope, right_inflection, right_slope)
    plt.plot(xs, y)
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Sigmoid Product")
    height = 1
    left_inflection = 10
    left_slope = 5
    right_inflection = 50
    right_slope = 0.2
    y = sigmoid_product(xs, height, left_inflection, left_slope, right_inflection, right_slope)
    plt.plot(xs, y)
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Spike")
    height = 1
    widt = 10
    center = 50
    y = spike(xs, height, widt, center)
    plt.plot(xs, y)
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Concave")
    height = 1
    inflection = 10
    end = 50
    y = concave(xs, height, inflection, end)
    plt.plot(xs, y)
    inflection = 60
    y = concave(xs, height, inflection, end)
    plt.plot(xs, y)
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Sigmoid")
    height = 1
    slope = 1
    inflection = 50
    y = sigmoid(xs, height, slope, inflection)
    plt.plot(xs, y)

    slope = -1
    inflection = 50
    y = sigmoid(xs, height, slope, inflection)
    plt.plot(xs, y)
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("S Shape")
    height = 1
    start = 30
    end = 70
    y = s_shape(xs, height, start, end)
    plt.plot(xs, y)
    plt.grid()
    plt.show()

    plt.figure()
    plt.title("Z Shape")
    height = 1
    start = 30
    end = 70
    y = z_shape(xs, height, start, end)
    plt.plot(xs, y)
    plt.grid()
    plt.show()

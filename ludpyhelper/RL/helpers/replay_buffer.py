import numpy as np
import random

class ER():
    def __init__(self, max_buffer_length=None):
        self.buffer = []
        self.max_buffer_length = max_buffer_length

    def append(self, experience, resize=True):
        self.buffer.append(experience)
        if resize:
            self._resize_buffer()

    def _resize_buffer(self):
        self.buffer = self._resize_list(self.buffer, self.max_buffer_length)

    def _resize_list(self, list, max_lenght):
        if max_lenght is not None and len(list) > max_lenght:
            list = list[-max_lenght:]
        return list

    def sample(self, batch_size):
        real_batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, real_batch_size)
        return np.array(batch)

    def __len__(self):
        return len(self.buffer)


# https://www.padl.ws/papers/Paper%2018.pdf
class PER(ER):
    def __init__(self, max_buffer_length=None):
        super().__init__(max_buffer_length)
        self.td_buffer = []

    def _resize_buffer(self):
        super()._resize_buffer()
        self.td_buffer = self._resize_list(self.td_buffer, self.max_buffer_length)

    def append(self, experience, td, resize=True):
        self.buffer.append(experience)
        self.td_buffer.append(td)

        if resize is True:
            self._resize_buffer()
        elif isinstance(resize, int):
            self.max_buffer_length = resize
            self._resize_buffer()

    def sample(self, batch_size, flat_sample=False):
        if not flat_sample:
            real_batch_size = min(batch_size, len(self.buffer))

            td_abs = np.abs(self.td_buffer)
            td_p = td_abs / td_abs.sum()

            sample_idxs = np.random.choice(len(self.buffer), size=real_batch_size, p=td_p)

            batch = np.array(self.buffer)[sample_idxs]
        else:
            batch = super().sample(batch_size)
        return batch


class AER(PER):
    def __init__(self, initial_memory_size, max_memory_size, n_old, k, shrinking_threshold=None, adaptively=True):
        super().__init__(max_buffer_length=initial_memory_size)
        self.delta_old = 0
        self.n_old = n_old
        self.k = k
        self.time_step = 0
        self.shrinking_threshold = shrinking_threshold
        self.max_memory_size = max_memory_size
        self.adaptively = adaptively

    def append(self, experience, td, resize=True):
        self.buffer.append(experience)
        self.td_buffer.append(td)
        if resize is True:
            self._resize_buffer()
        elif isinstance(resize, int):
            self.max_buffer_length = resize
            super()._resize_buffer()


    def _resize_buffer(self):
        self.time_step += 1

        if self.adaptively and self.time_step % self.k == 0 and len(self.buffer) >= self.max_buffer_length:
            td_np = np.array(self.td_buffer)

            if self.shrinking_threshold is None:
                shrinking_threshold = td_np.mean()
            else:
                shrinking_threshold = self.shrinking_threshold

            delta_old_mark = np.sum(td_np[0:self.n_old])
            if (delta_old_mark > self.delta_old or self.max_buffer_length == self.k) and self.max_buffer_length <= self.max_memory_size:
                self.max_buffer_length += self.k
                self.delta_old = delta_old_mark
            elif delta_old_mark < (self.delta_old - shrinking_threshold):
                self.max_buffer_length -= self.k
                self.delta_old = np.sum(td_np[self.k:self.k + self.n_old])

        super()._resize_buffer()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    er = ER()

    er_len = []
    for i in range(10_000):
        er.append(i)
        er_len.append(len(er))

    er_max = ER(max_buffer_length=1000)

    er_max_len = []
    for i in range(10_000):
        er_max.append(i)
        er_max_len.append(len(er_max))

    per = PER(max_buffer_length=1000)

    per_len = []
    for i in range(10_000):
        per.append(i, i, False)
        per_len.append(len(per))

    per_samples = []
    for _ in range(1_000):
        per_samples.append(per.sample(10, flat_sample=True))
    per_samples = np.array(per_samples).flatten()

    aer = AER(initial_memory_size = 200, max_memory_size=10_000,
              n_old = 100, k = 20, shrinking_threshold=0, adaptively=True)
    tdhist = np.load(r"C:\Users\simon\OneDrive - Syddansk Universitet\Studie\8 semester\ToolsAI\AIOpgaven\tdhist.npy")
    aer_len = []
    for i, td in enumerate(tdhist):
        aer.append(i, td)
        aer_len.append(len(aer))


    plt.figure()
    plt.plot(er_len, label="er_len")
    plt.plot(er_max_len, label="er_max_len")
    plt.plot(per_len, label="per_len")
    plt.plot(aer_len, label="aer_len")
    plt.legend()
    plt.show()

    plt.figure()
    plt.hist(per_samples)
    plt.show()
    pass

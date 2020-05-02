import threading
import queue
import time
from tqdm import tqdm
import os


class WorkerThread(threading.Thread):
    def __init__(self, task_queue, result_queue, sleep_time=0.1, kill_when_done=False):
        threading.Thread.__init__(self)

        self.task_queue = task_queue
        self.result_queue = result_queue

        self._kill = False
        self.kill_when_done = kill_when_done

        self.sleep_time = sleep_time

    def set_sleep_time(self, sleep_time):
        self.sleep_time = sleep_time

    def set_kill_when_done(self, kill_when_done):
        self.kill_when_done = kill_when_done

    def kill(self):
        self._kill = True

    def run(self):
        while not self._kill:
            if self.task_queue.qsize() > 0:
                func, arg = self.task_queue.get()
                res = func(**arg)
                self.result_queue.put(res)

                self.task_queue.task_done()

            if self.kill_when_done and self.task_queue.qsize():
                self._kill = True
            else:
                time.sleep(self.sleep_time)


class ThreadHandler:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.out_stading_jobs = 0
        self.hreads = []

    def make_threads(self, n, fcpu=None, sleep_time=0.1, kill_when_done=False):
        if fcpu is not None:
            cpu_count = os.cpu_count()
            n = round(cpu_count * fcpu)

        for _ in range(n):
            t = WorkerThread(self.task_queue, self.result_queue, sleep_time, kill_when_done)
            self.hreads.append(t)

    def start_threads(self):
        for t in self.hreads:
            t.start()

    def put_job(self, func, args):
        self.task_queue.put([func, args])
        self.out_stading_jobs += 1

    def available_results(self):
        return self.result_queue.qsize()

    def get_result(self):
        self.out_stading_jobs -= 1
        return self.result_queue.get()

    def all_jobs_done(self):
        return self.task_queue.all_tasks_done

    def join(self, force=False):
        if not force:
            self.task_queue.join()

        # First give kill sign
        for t in self.hreads:
            t.kill()
        # Then wait for all to join
        for t in self.hreads:
            t.join()

    def threads_alive(self):
        alive = [t.is_alive() for t in self.hreads]
        n_alive = sum(alive)
        return n_alive

    def progress_bar(self, bar_sleep=0.1):
        # TODO: Do so it can handel error in the threds

        pro_bar = tqdm(total=self.out_stading_jobs)

        qs_old = self.result_queue.qsize()
        qs_new = self.result_queue.qsize()

        n_alive = self.threads_alive()

        while qs_new < self.out_stading_jobs and n_alive > 0:
            qs_new = self.result_queue.qsize()

            if qs_new != qs_old:
                pro_bar.update(qs_new - qs_old)
            qs_old = qs_new

            n_alive = self.threads_alive()

            time.sleep(bar_sleep)
import time
from collections import deque


class FpsTracker:
    def __init__(self, window_size: int = 60):
        # stores the last `window_size` frame durations for the rolling average.
        self.frame_times = deque(maxlen=window_size)
        # records the timestamp of the initialization.
        self.last_tick = time.perf_counter()

    def tick(self) -> float:
        # calculates the time elapsed since the last tick.
        current_time = time.perf_counter()
        delta = current_time - self.last_tick
        self.frame_times.append(delta)
        self.last_tick = current_time

        # computes and returns the smoothed frames per second.
        return 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0.0

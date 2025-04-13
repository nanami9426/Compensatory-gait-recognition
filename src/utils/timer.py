import time

class Timer:
    def __init__(self):
        self._duration = 0
        self.s = None

    def start(self):
        self.s = time.time()

    def stop(self):
        assert self.s is not None
        self._duration += time.time() - self.s
        self.s = None

    def clear(self):
        self._duration = 0

    @property
    def t(self):
        return round(self._duration, 2)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

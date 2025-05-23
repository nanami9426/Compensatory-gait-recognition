import torch


class Window:
    def __init__(self, device, window_size, frame_shape):
        self._data = torch.zeros(window_size, *frame_shape, device=device)
        self.window_size = window_size
        self.current_size = 0

    def add(self, x):
        assert self.current_size < self.window_size
        self._data[self.current_size, :, :] = x
        self.current_size += 1
        if self.current_size == self.window_size:
            return True
        return False

    def clear(self):
        self._data.zero_()
        self.current_size = 0

    @property
    def data(self):
        return self._data.clone().detach()

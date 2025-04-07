import torch


class Window:
    def __init__(self, device, window_size, frame_shape):
        self.data = torch.zeros(window_size, *frame_shape, device=device)
        self.window_size = window_size
        self.current_size = 0

    def add(self, x):
        assert self.current_size < self.window_size
        self.data[self.current_size, :, :] = x
        self.current_size += 1
        if self.current_size == self.window_size:
            return True
        return False

    def clear(self):
        self.data[:] = 0
        self.current_size = 0


if __name__ == '__main__':
    window = Window('cuda:0', 4, (17, 2))
    while True:
        x = torch.randn(1, 17, 2)
        flag = window.add(x)
        if flag:
            break
    print(window.data[0, 0, :], window.data.shape)
    window.clear()
    print(window.data[0, 0, :], window.data.shape)
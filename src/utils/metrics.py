import time
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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


def accuracy(y_hat, y):
    if len(y_hat[0]) > 1 and len(y_hat.shape) > 1:
        y_hat = y_hat.argmax(1)
    cmp = y == y_hat.type(y.dtype)
    return cmp.type(y.dtype).sum().float()


def evaluate_accuracy(net, data_iter, device=None, batch_first=False):
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    for X, y in data_iter:
        if batch_first:
            X = X.to(device)
        else:
            X = X.permute(1, 0, 2).to(device)
        y = y.to(device)
        pred = net(X)
        metric.add(accuracy(pred, y), y.numel())
    return metric[0] / metric[1]


def evaluate_metrics(net, data_iter, device=None, batch_first=False, average='macro'):
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device

    all_preds = []
    all_labels = []

    for X, y in data_iter:
        if batch_first:
            X = X.to(device)
        else:
            X = X.permute(1, 0, 2).to(device)
        y = y.to(device)

        with torch.no_grad():
            pred = net(X)
            if pred.dim() > 1 and pred.size(1) > 1:
                pred = pred.argmax(dim=1)
            else:
                pred = (pred > 0.5).long()  # 二分类时用阈值判断

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=average, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=average, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=average, zero_division=0)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
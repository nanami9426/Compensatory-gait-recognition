from utils import metrics
import torch
from torch import nn
import os
import pandas as pd
from fastapi import Response, APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from nets.net import PoseNet, LightTransformer, PoseTextCNN
from conf import window_size, hidden_size, num_layers, kernel_sizes, nums_channels
from pathlib import Path
from datetime import datetime

train_router = APIRouter()
device = torch.device("cuda:0")


def train(net, train_iter, test_iter=None, num_epochs=16, devices=[torch.device("cuda:0")], batch_first=False):
    optimizer = torch.optim.Adam(net.parameters())
    loss = nn.CrossEntropyLoss(reduction='none')
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    timer = metrics.Timer()
    test_acc = None
    test_metrics = None
    for epoch in range(num_epochs):
        metric = metrics.Accumulator(4)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            net.train()
            if batch_first:
                X = X.to(devices[0])
            else:
                X = X.to(devices[0]).permute(1, 0, 2)
            y = y.to(devices[0])
            optimizer.zero_grad()
            pred = net(X)
            l = loss(pred, y)
            l.sum().backward()
            optimizer.step()
            train_acc = metrics.accuracy(pred, y)
            metric.add(l.sum(), train_acc, y.shape[0], y.numel())
            timer.stop()
        if test_iter is not None:
            # test_acc = metrics.evaluate_accuracy(net, test_iter,batch_first=batch_first)
            test_metrics = metrics.evaluate_metrics(net, test_iter,batch_first=batch_first)
        yield f'Epoch: {epoch + 1}\n loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[3]:.3f} \n'
        if test_metrics is not None:
            yield f' test acc {test_metrics["accuracy"]:.3f}, precision {test_metrics["precision"]:.3f}, recall {test_metrics["recall"]:.3f}, f1 {test_metrics["f1"]:.3f} \n'
    yield f'{metric[2] * num_epochs / timer.t:.1f} examples/sec on {str(devices)} \n'

    now = datetime.now()
    time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    net_path = os.path.join(Path(__file__).resolve().parents[2], f"models\{net.module.name}_{time_str}.pth")
    torch.save(net.module.state_dict(), net_path)
    yield f'Net is saved in {net_path} \n'


class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, window_size):
        self.root_dir = root_dir
        self.data = []  # (path, label, start_idx)
        self.window_size = window_size
        for fname in os.listdir(root_dir):
            if fname.endswith(".csv"):
                label = int(fname.split("_")[-1][0])
                path = os.path.join(root_dir, fname)
                df = pd.read_csv(path, header=None).iloc[1:, 1:]
                total_len = df.shape[0]
                num_windows = total_len // window_size
                for i in range(num_windows):
                    self.data.append((path, label, i * window_size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label, start_idx = self.data[idx]
        df = pd.read_csv(path, header=None).iloc[1:, 1:]
        segment = df.iloc[start_idx:start_idx + self.window_size].values
        return torch.tensor(segment, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def get_data_iter(batch_size=32, is_train=True):
    if is_train:
        raw_data_path = os.path.join(Path(__file__).resolve().parents[2], "preprocess/rawdata/train")
    else:
        raw_data_path = os.path.join(Path(__file__).resolve().parents[2], "preprocess/rawdata/test")
    dataset = PoseDataset(raw_data_path, 24)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return data_loader


def get_net(net_name):
    if net_name == 'pose_net':
        return PoseNet(window_size, hidden_size, num_layers), False
    elif net_name == "pose_text_cnn":
        return PoseTextCNN(kernel_sizes,nums_channels), True
    elif net_name == "pose_transformer":
        return LightTransformer(), True
    return None, None


class TrainRequest(BaseModel):
    batch_size: int
    evaluate_test: bool
    net_name: str
    num_epochs: int


@train_router.post("/func/train")
def train_api(params: TrainRequest):
    batch_size = params.batch_size
    evaluate_test = params.evaluate_test
    net_name = params.net_name
    num_epochs = params.num_epochs
    net, batch_first = get_net(net_name)
    net.to(device)
    train_iter = get_data_iter(batch_size, True)
    if evaluate_test:
        test_iter = get_data_iter(batch_size, False)
    else:
        test_iter = None
    generator = train(net, train_iter, test_iter, num_epochs, devices=[torch.device("cuda:0")], batch_first=batch_first)
    return StreamingResponse(generator, media_type='text/plain')

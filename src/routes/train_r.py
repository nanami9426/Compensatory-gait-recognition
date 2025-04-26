from utils import metrics
import torch
from torch import nn
import os
import pandas as pd
from fastapi import Response, APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from nets.net import Pnet
from conf import window_size, hidden_size, num_layers
from pathlib import Path

train_router = APIRouter()

def train(net, train_iter, test_iter=None, num_epochs=16, devices=[torch.device("cuda:0")], save_name="testnet_01.pth"):
    optimizer = torch.optim.Adam(net.parameters())
    loss = nn.CrossEntropyLoss(reduction='none')
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    # num_batches = len(train_iter)
    timer = metrics.Timer()
    test_acc = None
    for epoch in range(num_epochs):
        metric = metrics.Accumulator(4)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            net.train()
            X, y = X.to(devices[0]).permute(1, 0, 2), y.to(devices[0])
            optimizer.zero_grad()
            pred = net(X)
            l = loss(pred, y)
            l.sum().backward()
            optimizer.step()
            train_acc = metrics.accuracy(pred, y)
            metric.add(l.sum(), train_acc, y.shape[0], y.numel())
            timer.stop()
        if test_iter is not None:
            test_acc = metrics.evaluate_accuracy(net, test_iter)
        yield f'Epoch: {epoch+1}, loss {metric[0] / metric[2]:.3f}, train acc {metric[1] / metric[3]:.3f} \n'
        if test_acc is not None:
            yield f'test acc {test_acc:.3f} \n'
    yield f'{metric[2] * num_epochs / timer.t:.1f} examples/sec on {str(devices)} \n'
    path = os.path.join(Path(__file__).resolve().parents[2], f"models\{save_name}")
    yield f'net saved at {path} \n'

class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, window_size):
        self.root_dir = root_dir
        self.data = [] # (path, label, start_idx)
        self.window_size = window_size
        for fname in os.listdir(root_dir):
            if fname.endswith(".csv"):
                label = int(fname.split("_")[-1][0])
                path = os.path.join(root_dir, fname)
                df = pd.read_csv(path, header=None).iloc[1:, 1:]
                total_len = df.shape[0]
                num_windows = total_len // window_size
                for i in range(num_windows):
                    self.data.append((path, label, i*window_size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label, start_idx = self.data[idx]
        df = pd.read_csv(path, header=None).iloc[1:, 1:]
        segment = df.iloc[start_idx:start_idx+self.window_size].values
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
    if net_name == 'pnet':
        return Pnet(window_size, hidden_size, num_layers)

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
    device = torch.device("cuda:0")
    net = get_net(net_name).to(device)
    train_iter = get_data_iter(batch_size, True)
    if evaluate_test:
        test_iter = get_data_iter(batch_size, False)
    else:    
        test_iter = None
    generator = train(net, train_iter, test_iter, num_epochs, devices=[torch.device("cuda:0")])
    return StreamingResponse(generator, media_type='text/plain')
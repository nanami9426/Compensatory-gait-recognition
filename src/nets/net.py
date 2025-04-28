import torch
from torch import nn
from conf import window_size, hidden_size, num_layers


device = torch.device("cuda:0")

# InputShape([window_size, 17, 2]) 
# 批量：1，帧数：window_size，输入维度：17*2
class Pnet(nn.Module):
    def __init__(self, window_size, num_hiddens, num_layers):
        super().__init__()
        self.window_size = window_size
        self.encoder = nn.LSTM(17 * 2, num_hiddens, num_layers=num_layers, bidirectional=False)
        self.decoder = nn.Linear(num_hiddens, 2)
        self.name = 'pnet'
    def forward(self, x):
        # x = x.reshape(self.window_size, -1).unsqueeze(1)
        _, state = self.encoder(x)
        res = self.decoder(state[0][-1])
        return res







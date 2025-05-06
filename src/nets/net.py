import torch
from torch import nn
from conf import window_size, hidden_size, num_layers

device = torch.device("cuda:0")

# Input Shape([window_size, batch_size, 17*2])
class PoseNet(nn.Module):
    def __init__(self, window_size, num_hiddens, num_layers):
        super().__init__()
        self.window_size = window_size
        self.encoder = nn.LSTM(17 * 2, num_hiddens, num_layers=num_layers, bidirectional=False)
        self.decoder = nn.Linear(num_hiddens, 2)
        self.name = 'pose_net'
    def forward(self, x):
        _, state = self.encoder(x)
        res = self.decoder(state[0][-1])
        return res
    
# Input Shape([batch_size, window_size, 17*2])
class PoseTextCNN(nn.Module):
    def __init__(self, kernel_sizes, num_channels):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        self.name = 'pose_text_cnn'
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d((17*2), c, k))

    def forward(self, X):
        # X: Shape([batch_size, 24, 34])
        X = X.permute(0, 2, 1)
        encoding = torch.cat([
            self.relu(self.pool(conv(X)).squeeze(-1))
            for conv in self.convs
        ], dim=-1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs

# Input Shape([batch_size, window_size, 17*2])
class LightTransformer(nn.Module):
    def __init__(self, input_dim=17*2, embed_dim=64, num_heads=4, num_classes=2, ff_dim=128, window_size=24, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)  # [batch, window, 17*2] -> [batch, window, embed_dim]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.name = 'pose_transformer'
        self.pos_encoding = self.pos_enc(window_size, embed_dim)  # [1, window_size, embed_dim]

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x) + self.pos_encoding
        x = x.permute(0, 2, 1)
        x = self.pooling(x).squeeze(-1)
        out = self.classifier(x)
        return out

    def pos_enc(self, window_size, embed_dim):
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim 必须是偶数")
        position = torch.arange(window_size, dtype=torch.float32).unsqueeze(1)  # (window_size, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))  # (embed_dim/2,)
        pe = torch.zeros(window_size, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        pe = pe.unsqueeze(0)  # (1, window_size, embed_dim)
        return nn.Parameter(pe, requires_grad=True)



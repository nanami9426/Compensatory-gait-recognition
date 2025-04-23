import torch
import os
import pandas as pd

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
        return torch.tensor(segment), torch.tensor(label, dtype=torch.long)
import copy
import os

import numpy as np
import torch
from torch import nn

from src.stamina.config import RANDOM_SEED


def set_seed() -> None:
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class LSTMTrainer:
    def __init__(self, model, device, criterion, optimizer):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def train_epoch(self, x_batch, y_batch):
        self.model.train()
        self.optimizer.zero_grad()
        pred = self.model(x_batch)
        loss = self.criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())

    def evaluate(self, x_data, y_data):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_data)
            loss = self.criterion(pred, y_data)
        return float(loss.item())


def build_model(input_dim: int):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster(input_size=input_dim, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    return model, device


def save_checkpoint(path, model, config):
    torch.save({"model_state_dict": model.state_dict(), "config": config}, path)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt

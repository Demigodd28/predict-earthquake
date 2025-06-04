# models/model.py
import torch.nn as nn

class EarthquakePredictor(nn.Module):
    def __init__(self, input_dim):
        super(EarthquakePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 輸出為機率
        )

    def forward(self, x):
        return self.model(x)

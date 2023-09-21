import torch
import torch.nn as nn


class StressNet(nn.Module):
    def __init__(self, layer_1_dim):
        super(StressNet, self).__init__()
        self.learning_rate = 1e-3
        self.num_epochs = 100
        self.criterion = nn.CrossEntropyLoss()
        self.train_batch_size = 64
        self.test_batch_size = 32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fc = nn.Sequential(
            nn.Linear(layer_1_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        return self.fc(x)

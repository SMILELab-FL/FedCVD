import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers, num_classes, bidirectional=False):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_channels, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 转换数据维度 (batch, length, channels) -> (batch, channels, length)
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 只取最后一个时间步的输出
        out = out[:, -1, :]

        # 全连接层
        out = self.fc(out)

        # 使用sigmoid激活函数得到每个类别的概率
        out = self.sigmoid(out)

        return out

def lstm(input_channels=12, hidden_size=256, num_layers=2, num_classes=20, bidirectional=False):
    return LSTM(input_channels, hidden_size, num_layers, num_classes, bidirectional)
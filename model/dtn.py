import math
from torch.nn import Module
import torch


class PositionalEncoding(Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CNNBlock(Module):
    def __init__(self, input_channels):
        super(CNNBlock, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_channels, 128, kernel_size=14, stride=3, padding=2)
        self.conv2 = torch.nn.Conv1d(128, 256, kernel_size=14, stride=3, padding=0)
        self.conv3 = torch.nn.Conv1d(256, 256, kernel_size=10, stride=2, padding=0)
        self.conv4 = torch.nn.Conv1d(256, 256, kernel_size=10, stride=2, padding=0)
        self.conv5 = torch.nn.Conv1d(256, 256, kernel_size=10, stride=1, padding=0)
        self.conv6 = torch.nn.Conv1d(256, 256, kernel_size=10, stride=1, padding=0)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        y = self.activation(self.conv1(x))
        y = self.activation(self.conv2(y))
        y = self.activation(self.conv3(y))
        y = self.activation(self.conv4(y))
        y = self.activation(self.conv5(y))
        y = self.activation(self.conv6(y))
        return y


class TransformerBlock(Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=8, dim_feedforward=2048, dropout=0.1, batch_first=True, ddeep=64, dclass=20):
        super(TransformerBlock, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=batch_first
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_encoder_layers
        )
        self.global_pooling = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(d_model, ddeep)
        self.fc2 = torch.nn.Linear(ddeep, dclass)

    def forward(self, x):
        y = self.positional_encoding(x)
        y = self.transformer_encoder(y)
        y = y.transpose(1, 2)
        y = self.global_pooling(y).squeeze(-1)
        y = torch.relu(self.fc(y))
        y = self.fc2(y)
        return y



class DeepTransformerNetwork(Module):
    def __init__(self, input_channels, d_model=256, nhead=8, num_encoder_layers=8, dim_feedforward=2048, dropout=0.1, ddeep=64, dclass=20):
        super(DeepTransformerNetwork, self).__init__()
        self.cnn_block = CNNBlock(input_channels)
        self.transformer_block = TransformerBlock(
            d_model, nhead, num_encoder_layers, dim_feedforward, dropout, ddeep, dclass
        )
        self.act = torch.nn.Sigmoid()

        # init with xavier
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        y = self.cnn_block(x)
        y = y.transpose(1, 2)
        y = self.transformer_block(y)
        y = self.act(y)
        return y


def dtn(input_channels=12, d_model=256, nhead=8, num_encoder_layers=8, dim_feedforward=2048, dropout=0.1, ddeep=64, dclass=20):
    return DeepTransformerNetwork(input_channels, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, ddeep, dclass)



if __name__ == "__main__":
    sample = torch.randn((8, 12, 5000))
    model = DeepTransformerNetwork(12)
    out = model(sample)
    print(out.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout_rate=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)  # position-wise
        self.w_2 = nn.Linear(d_hidden, d_model)  # position-wise
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x
    

if __name__ == "__main__":
    positionWise = PositionWiseFeedForward(512, 2048)
    print(positionWise)
    x = torch.randn(2, 4, 512)
    print(positionWise(x).shape)

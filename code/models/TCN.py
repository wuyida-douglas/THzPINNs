import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNRegressor(nn.Module):
    def __init__(self, input_length, num_outputs=16, num_channels=16, kernel_size=7, dropout=0):
        super().__init__()
        N = 7  # 固定为7层
        channels = [num_channels] * N
        layers = []
        in_channels = 1
        for i, out_channels in enumerate(channels):
            dilation_size = 2 ** i
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size, padding=(kernel_size-1)*dilation_size, dropout=dropout))
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)
        # QKV自注意力池化参数
        self.q_proj = nn.Linear(num_channels, num_channels)
        self.k_proj = nn.Linear(num_channels, num_channels)
        self.v_proj = nn.Linear(num_channels, num_channels)
        self.fc = nn.Linear(num_channels, num_outputs)
        self.tanh = nn.Tanh()
        self.param_min = torch.tensor([
            1,  0.001 ,  0.001 , 0.001   ,   # 第一层材料参数
            1,  0.001 ,  0.001 , 0.001   ,
            1,  0.001 ,  0.001 , 0.001   ,    # 第三层材料参数
            2, 3, 25, 30         # 厚度
        ], dtype=torch.float32)
        self.param_max = torch.tensor([
            6,  500,  500, 500   ,   # 第一层材料参数
            6,  500,  500, 500   ,
            6,  500,  500, 500   ,   # 第三层材料参数
            50, 35, 60, 200        # 厚度
        ], dtype=torch.float32)
        print(f"TCN层数: {N}, 感受野: {1 + (kernel_size-1)*(2**N - 1)}")

    def forward(self, x):
        # x: [batch, 1, 信号长度]
        y = self.tcn(x)  # [batch, channels, L]
        y = y.permute(0, 2, 1)  # [batch, L, channels]
        Q = self.q_proj(y)  # [batch, L, channels]
        K = self.k_proj(y)  # [batch, L, channels]
        V = self.v_proj(y)  # [batch, L, channels]
        attn_score = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)  # [batch, L, L]
        attn_weight = torch.softmax(attn_score, dim=-1)  # [batch, L, L]
        attn_out = torch.matmul(attn_weight, V)  # [batch, L, channels]
        pooled = attn_out.mean(dim=1)  # [batch, channels]，全局池化
        net_out = self.fc(pooled)  # [batch, 16]
        param_min = self.param_min.to(net_out.device)
        param_max = self.param_max.to(net_out.device)
        params = (self.tanh(net_out)+1)/2 * (param_max - param_min) + param_min  # [batch, 16]
        return params 
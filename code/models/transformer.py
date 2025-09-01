import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))  # 可学习的位置编码
    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1), :]

class SimpleTransformerRegressor(nn.Module):
    def __init__(self, input_length, num_outputs=16, d_model=32, nhead=4, num_layers=3, dropout=0):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)  # [batch, L, 1] -> [batch, L, d_model]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # 可学习CLS token
        self.pos_encoder = PositionalEncoding(input_length + 1, d_model)  # 注意长度+1
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_hidden = nn.Linear(d_model, 64)
        self.fc = nn.Linear(64, num_outputs)
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
    def forward(self, x):
        # x: [batch, 1, L]
        x = x.permute(0, 2, 1)  # [batch, L, 1]
        x = self.input_proj(x)   # [batch, L, d_model]
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, d_model]
        x = torch.cat((cls_tokens, x), dim=1)  # [batch, L+1, d_model]
        x = self.pos_encoder(x)  # 加可学习位置编码
        x = self.transformer(x)  # [batch, L+1, d_model]
        x = x[:, 0, :] # 只取CLS token输出 [batch, d_model]
        x = self.fc_hidden(x)    # [batch, 64]
        x = self.tanh(x)
        net_out = self.fc(x)     # [batch, 16]
        param_min = self.param_min.to(net_out.device)
        param_max = self.param_max.to(net_out.device)
        params = (self.tanh(net_out)+1)/2 * (param_max - param_min) + param_min
        return params 
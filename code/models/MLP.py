import torch
import torch.nn as nn



class SimpleMLP(nn.Module):
    def __init__(self, input_length, dropout=0.1):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc0 = nn.Linear(input_length, 256)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 16)
        self.tanh = nn.Tanh()

        #硬约束，也可以采用软约束
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
        # x: [batch, 1, 信号长度]
        x = self.flatten(x)  # [batch, 信号长度]
        x = self.tanh(self.fc0(x))
        x = self.tanh(self.fc1(x))
        net_out = self.fc2(x)  # [batch, 16]
        param_min = self.param_min.to(net_out.device)
        param_max = self.param_max.to(net_out.device)
        params = (torch.tanh(net_out)+1)/2 * (param_max - param_min) + param_min  # [batch, 16]
        return params
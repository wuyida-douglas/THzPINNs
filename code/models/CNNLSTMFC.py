import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.Tanh()
        self.need_proj = (in_channels != out_channels) or (stride != 1)
        if self.need_proj:
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.proj = None
        # Xavier初始化（适用于Tanh）
        nn.init.xavier_uniform_(self.conv.weight)
        if self.need_proj and self.proj is not None:
            nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        if self.proj is not None:
            identity = self.proj(identity)
        out = out + identity
        out = self.relu(out)
        return out

class CNNLSTMFC(nn.Module):
    def __init__(self):
        super(CNNLSTMFC, self).__init__()
        self.res_block1 = ResidualBlock(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.res_block2 = ResidualBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.res_block3 = ResidualBlock(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.res_block4 = ResidualBlock(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.res_block5 = ResidualBlock(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.res_block6 = ResidualBlock(256, 512, kernel_size=3, stride=1, padding=1)
        self.pool6 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.res_block7 = ResidualBlock(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool7 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.res_block8 = ResidualBlock(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool8 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.res_block9 = ResidualBlock(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool9 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)  # 恢复输出16个参数
        self.dropout = nn.Dropout(0.05)  # 新增Dropout层

        # # 恢复16维param_min和param_max
        # self.param_min = torch.tensor([
        #     1,  0.001  ,  0.1 , 0.01  ,   # 第一层材料参数
        #     1,  0.001 ,  0.1 , 0.01   ,
        #     1,  0.001 ,  0.1 , 0.01   ,    # 第三层材料参数
        #     2, 3, 25, 30         # 厚度
        # ], dtype=torch.float32)
        # self.param_max = torch.tensor([
        #     3.5,  3,  3, 2   ,   # 第一层材料参数
        #     3.5,  3,  3, 2   ,
        #     3.5,  3,  3, 2   ,   # 第三层材料参数
        #     50, 35, 60, 200        # 厚度
        # ], dtype=torch.float32)
                # 恢复16维param_min和param_max
        self.param_min = torch.tensor([
            1,  0.001  ,  0.001 , 0.001  ,   # 第一层材料参数
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

        self.relu = nn.Tanh()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        # # ====== 设置fc3.bias使输出初始值为指定参数 ======
        # desired_params = torch.tensor([
        #     2.8, 250, 250, 250,
        #     2.8, 250, 250, 250,
        #     2.8, 250, 250, 250,
        #     25, 17, 30, 70
        # ], dtype=torch.float32)
        # param_min = self.param_min
        # param_max = self.param_max
        # # 反推fc3输出前的值x: x = arctanh(2*(desired-param_min)/(param_max-param_min)-1)
        # x = torch.atanh(2 * (desired_params - param_min) / (param_max - param_min) - 1)
        # with torch.no_grad():
        #     self.fc3.bias.copy_(x)

    def forward(self, x):
        x = self.res_block1(x)
        x = self.pool1(x)
        x = self.res_block2(x)
        x = self.pool2(x)
        x = self.res_block3(x)
        x = self.pool3(x)
        x = self.res_block4(x)
        x = self.pool4(x)
        x = self.res_block5(x)
        x = self.pool5(x)
        x = self.res_block6(x)
        x = self.pool6(x)
        x = self.res_block7(x)
        x = self.pool7(x)
        x = self.res_block8(x)
        x = self.pool8(x)
        x = self.res_block9(x)
        x = self.pool9(x)
        x = x.permute(0, 2, 1)  # (batch, seq, feature)
        x, _ = self.lstm(x)
        # print("LSTM输出shape:", x.shape)
        # print("最后一个时间步前向输出均值:", x[:, -1, :x.shape[2]//2].mean().item())
        # print("最后一个时间步反向输出均值:", x[:, -1, x.shape[2]//2:].mean().item())
        # x = x[:, -1, :]
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Dropout after fc1
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)  # Dropout after fc2
        net_out = self.fc3(x)  # [batch, 16]
        # 物理约束
        param_min = self.param_min.to(net_out.device)
        param_max = self.param_max.to(net_out.device)
        params = (torch.tanh(net_out)+1)/2 * (param_max - param_min) + param_min  # [batch, 16]
        return params

    def print_gradients(self):
        print("\n每一层参数的梯度范数如下：")
        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad norm = {param.grad.norm().item():.6e}")
            else:
                print(f"{name}: grad is None")
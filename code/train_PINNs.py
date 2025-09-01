import numpy as np
import torch
import torch.optim as optim
import random
from physics_torch import simulate_signal_torch
import os
import torch.utils.data as data
from torch.utils.tensorboard.writer import SummaryWriter
from models.MLP import SimpleMLP
import time



# ====== 配置参数集中管理 ======
config = {
    'BATCH_SIZE': 256,
    'scale': 1/(2.2e4),
    'w1': 1.0,
    'w2': 1.2,
    'w3': 1.0,
    'lambda_supervise': 8e-6,
    'DATA_ROOT': r'./data/训练集_0707',
    'VAL_DATA_ROOT': r'./data/验证集_0727',
    'input_length': 3451,
    'learning_rate': 1e-4,
    'weight_decay': 0,
    'clip_value': 1.5e-1,
    'epochs': 500,
    'train_log_dir': r'/root/tf-logs/beijing/train',
    'val_log_dir': r'/root/tf-logs/beijing/val',
    'checkpoint_path': './checkpoints/SOTA模型_35.pth',
    'max_val_batches': 5,
    'num_workers': 8,
    'pin_memory': True,
    'dropout': 0,
    'lr_decay_gamma': 0.9985,  # 每个epoch指数衰减系数
}

# ====== 固定随机种子，保证可复现 ======
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def loss_func_torch(sim, real, N, ml1, ml2, ml3, ml4, plot_info=None):
    if sim.ndim == 1:
        sim = sim.unsqueeze(0)
        real = real.unsqueeze(0)
    B = sim.shape[0]
    device = sim.device
    sim_full_fft = torch.zeros((B, N), dtype=torch.complex128, device=device)
    sim_full_fft[:, ml1:ml2+1] = sim
    sim_full_fft[:, ml3:ml4+1] = torch.conj(torch.flip(sim, dims=[1]))
    sim_time = torch.fft.ifft(sim_full_fft, dim=1).real  # [B, N]

    real_full_fft = torch.zeros((B, N), dtype=torch.complex128, device=device)
    real_full_fft[:, ml1:ml2+1] = real
    real_full_fft[:, ml3:ml4+1] = torch.conj(torch.flip(real, dims=[1]))
    real_time = torch.fft.ifft(real_full_fft, dim=1).real  # [B, N]

    return torch.mean((sim_time - real_time) ** 2)

class PairedSignalDataset(data.Dataset):
    def __init__(self, root_dir, scale=1/(2.2e4)):
        self.pairs = []
        self.scale = scale
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('_1.txt'):
                    measure_path = os.path.join(subdir, file)
                    ref_path = measure_path.replace('_1.txt', '_2.txt')
                    if os.path.exists(ref_path):
                        self.pairs.append((measure_path, ref_path))
        print(f"共找到{len(self.pairs)}对实测-参考信号")
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        measure_path, ref_path = self.pairs[idx]
        measure_data = np.loadtxt(measure_path)
        ref_data = np.loadtxt(ref_path)
        time = torch.from_numpy(measure_data[:, 0]).float()
        measure = torch.from_numpy(measure_data[:, 1] * self.scale).float() #这里还好
        ref = torch.from_numpy(ref_data[:, 1] * self.scale).float() #这里还好
        # 解析厚度标签
        subfolder = os.path.basename(os.path.dirname(measure_path))
        try:
            thickness_label = [float(x) for x in subfolder.split('-')]
        except Exception as e:
            thickness_label = [0.0, 0.0, 0.0]  # 若解析失败，填0
            print("ERROR!!!!!")
        thickness_label = torch.tensor(thickness_label, dtype=torch.float32)
        return time, measure, ref, measure_path, thickness_label


# ====== 验证集评估函数 ======
def evaluate_on_val(model, val_dataloader, device, max_batches=None):
    model.eval()
    total_data_loss = 0.0
    total_thickness_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            measure_time_batch, measure_batch, ref_batch, measure_path_batch, thickness_label_batch = batch
            # 全部转为torch tensor并放到device
            measure_time_batch = measure_time_batch.to(device)
            measure_batch = measure_batch.to(device)
            ref_batch = ref_batch.to(device)
            thickness_label_batch = thickness_label_batch.to(device)
            # ==== 批量FFT/IFFT处理 ====
            ml1, ml2, ml3, ml4 = 2,1725,1726,3449 #5,240,3211,3446
            N = measure_time_batch.shape[1]
            dt_ps = (measure_time_batch[:, 1] - measure_time_batch[:, 0])  # shape: [BATCH_SIZE]
            dt_ps0 = dt_ps[0]
            N0 = N
            frequencies_Hz = torch.fft.fftfreq(N0, d=dt_ps0 * 1e-12, device=device)
            frequencies_THz = frequencies_Hz / 1e12
            frequencies_sel = frequencies_THz[ml1:ml2+1]
            # 批量FFT (torch实现)
            measure_fft = torch.fft.fft(measure_batch, dim=1)
            ref_fft = torch.fft.fft(ref_batch, dim=1)
            Fmeas_obj = measure_fft[:, ml1:ml2+1]
            Fref_obj = ref_fft[:, ml1:ml2+1]
            # 批量重组频谱
            ref_full_fft = torch.zeros_like(ref_fft, dtype=torch.complex128, device=device)
            measure_full_fft = torch.zeros_like(measure_fft, dtype=torch.complex128, device=device)
            ref_full_fft[:, ml1:ml2+1] = Fref_obj
            ref_full_fft[:, ml3:ml4+1] = torch.conj(torch.flip(Fref_obj, dims=[1]))
            measure_full_fft[:, ml1:ml2+1] = Fmeas_obj
            measure_full_fft[:, ml3:ml4+1] = torch.conj(torch.flip(Fmeas_obj, dims=[1]))
            # 批量IFFT (torch实现)
            # ref_filtered = torch.fft.ifft(ref_full_fft, dim=1).real.float()
            measure_filtered = torch.fft.ifft(measure_full_fft, dim=1).real.float()
            # 转为torch tensor
            measure_tensor = measure_filtered.unsqueeze(1)  # [BATCH_SIZE, 1, 信号长度]
            params_batch = model(measure_tensor)
            # 批量调用物理仿真和loss
            HW = simulate_signal_torch(params_batch, frequencies_sel)  # [BATCH_SIZE, N_freq]
            Fsim_obj = Fref_obj.clone().detach() * HW  # [BATCH_SIZE, N_freq]
            Fmeas_obj_tensor = Fmeas_obj.clone().detach()

            batch_loss = loss_func_torch(
                Fsim_obj,
                Fmeas_obj_tensor,
                N, ml1, ml2, ml3, ml4,
                plot_info=None
            )

            # 新增：厚度监督loss
            pred_thickness = params_batch[:, 12:15]  # [BATCH_SIZE, 3]
            label_thickness = thickness_label_batch.clone().detach()
            w1, w2, w3 = 1.0, 1.2, 1.0  # 可调
            thickness_loss = ((w1 * (pred_thickness[:,0] - label_thickness[:,0]) ** 2 +
                              w2 * (pred_thickness[:,1] - label_thickness[:,1]) ** 2 +
                              w3 * (pred_thickness[:,2] - label_thickness[:,2]) ** 2).mean()) * config['lambda_supervise']
            total_data_loss += batch_loss.item() * measure_tensor.shape[0]
            total_thickness_loss += thickness_loss.item() * measure_tensor.shape[0]
            count += measure_tensor.shape[0]
    avg_data_loss = total_data_loss / count
    avg_thickness_loss = total_thickness_loss / count
    return  avg_data_loss, avg_thickness_loss

# ====== 计算自适应权重（基于雅可比和核矩阵trace） ======
def compute_adaptive_weights(model, measure_tensor, label_thickness, signal_loss_fn, thickness_loss_fn):
    # 计算signal_loss和thickness_loss对参数的雅可比
    model.zero_grad()
    signal_loss = signal_loss_fn()
    print(signal_loss)
    grads_signal = torch.autograd.grad(signal_loss, [p for p in model.parameters() if p.requires_grad], retain_graph=True, create_graph=True)
    grads_signal = torch.cat([g.view(-1) for g in grads_signal if g is not None])
    K_signal = torch.sum(grads_signal ** 2)
    print(K_signal)

    thickness_loss = thickness_loss_fn()
    print(thickness_loss)
    grads_thickness = torch.autograd.grad(thickness_loss, [p for p in model.parameters() if p.requires_grad], retain_graph=True, create_graph=True)
    grads_thickness = torch.cat([g.view(-1) for g in grads_thickness if g is not None])
    K_thickness = torch.sum(grads_thickness ** 2)
    print(K_thickness)

    K_trace = K_signal + K_thickness + 1e-9
    w_signal = K_trace / (K_signal + 1e-9)
    w_thickness = K_trace / (K_thickness + 1e-9)

    # 归一化权重
    w_sum = w_signal + w_thickness
    w_signal = w_signal / w_sum
    w_thickness = w_thickness / w_sum

    print(f"[Adaptive Weights] w_signal={w_signal.item():.7f}, w_thickness={w_thickness.item():.7f}")
    return w_signal.detach(), w_thickness.detach()

# ====== 计算thickness_loss的函数，便于切换loss类型 ======
def thickness_loss_fn(pred, label, config):
    # MSE loss
    return ((config['w1'] * (pred[:,0] - label[:,0]) ** 2 +
             config['w2'] * (pred[:,1] - label[:,1]) ** 2 +
             config['w3'] * (pred[:,2] - label[:,2]) ** 2).mean()) * config['lambda_supervise']

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # ====== 构建Dataset和DataLoader ======
    dataset = PairedSignalDataset(config['DATA_ROOT'], scale=config['scale'])
    dataloader = data.DataLoader(dataset, batch_size=config['BATCH_SIZE'], shuffle=True, drop_last=True, num_workers=config['num_workers'], pin_memory=config['pin_memory'])

    # ====== 加载验证集 ======
    val_dataset = PairedSignalDataset(config['VAL_DATA_ROOT'], scale=config['scale'])
    val_dataloader = data.DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, drop_last=True, num_workers=config['num_workers'], pin_memory=config['pin_memory'])

    print(f"[DEBUG] len(dataset)={len(dataset)}, len(dataloader)={len(dataloader)}")
    if len(dataloader) == 0:
        print("[ERROR] DataLoader为空，请检查实测信号数量是否大于等于BATCH_SIZE且为其整数倍！")

    model = SimpleMLP(input_length = config['input_length'], dropout=config['dropout'])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_decay_gamma'])

    # ====== 初始化TensorBoard writer ======
    writer_train = SummaryWriter(config['train_log_dir'])
    writer_val = SummaryWriter(config['val_log_dir'])
    print("构建Dataset和DataLoader完毕")
    print('当前工作目录:', os.getcwd())

    CHECKPOINT_PATH = config['checkpoint_path']
    if os.path.exists(CHECKPOINT_PATH):
        print(f"检测到已有checkpoint，正在加载: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("模型和优化器参数已加载，继续训练。")
        else:
            model.load_state_dict(checkpoint)
            print("模型参数已加载，继续训练。")

    else:
        print("未检测到checkpoint，将从头开始训练。")

    for epoch in range(config['epochs']):
        if epoch == 0:
            start_time = time.time()  # 记录训练开始时间
        elapsed = time.time() - start_time
        print(f"\n========== 开始第 {epoch+1} 个 epoch ==========")
        print(f"[Epoch {epoch+1}] 当前学习率: {optimizer.param_groups[0]['lr']}")
        print(f"[Epoch {epoch+1}] 已用时: {elapsed:.2f} 秒")
        model.train()
        epoch_loss = 0.0

        # ====== 阶段2：引入信号loss和自适应权重 ======
        if(epoch > -1):
            for batch_idx, batch in enumerate(dataloader):
                measure_time_batch, measure_batch, ref_batch, measure_path_batch, thickness_label_batch = batch  # [BATCH_SIZE, 信号长度], [BATCH_SIZE]
                measure_time_batch = measure_time_batch.to(device)
                measure_batch = measure_batch.to(device)
                ref_batch = ref_batch.to(device)
                thickness_label_batch = thickness_label_batch.to(device)  # [BATCH_SIZE, 3]
                # ==== 批量FFT/IFFT处理 ====
                ml1, ml2, ml3, ml4 = 2,1725,1726,3449
                N = measure_time_batch.shape[1]
                dt_ps = (measure_time_batch[:, 1] - measure_time_batch[:, 0])  # shape: [BATCH_SIZE]
                dt_ps0 = dt_ps[0]
                N0 = N
                frequencies_Hz = torch.fft.fftfreq(N0, d=dt_ps0 * 1e-12, device=device)
                frequencies_THz = frequencies_Hz / 1e12
                frequencies_sel = frequencies_THz[ml1:ml2+1]
                # 批量FFT (torch实现)
                measure_fft = torch.fft.fft(measure_batch, dim=1)
                ref_fft = torch.fft.fft(ref_batch, dim=1)
                Fmeas_obj = measure_fft[:, ml1:ml2+1]
                Fref_obj = ref_fft[:, ml1:ml2+1]
                # 批量重组频谱
                ref_full_fft = torch.zeros_like(ref_fft, dtype=torch.complex128, device=device)
                measure_full_fft = torch.zeros_like(measure_fft, dtype=torch.complex128, device=device)
                ref_full_fft[:, ml1:ml2+1] = Fref_obj
                ref_full_fft[:, ml3:ml4+1] = torch.conj(torch.flip(Fref_obj, dims=[1]))
                measure_full_fft[:, ml1:ml2+1] = Fmeas_obj
                measure_full_fft[:, ml3:ml4+1] = torch.conj(torch.flip(Fmeas_obj, dims=[1]))
                # 批量IFFT (torch实现)
                # ref_filtered = torch.fft.ifft(ref_full_fft, dim=1).real.float()
                measure_filtered = torch.fft.ifft(measure_full_fft, dim=1).real.float()
                # 转为torch tensor
                measure_tensor = measure_filtered.unsqueeze(1)  # [BATCH_SIZE, 1, 信号长度]
                params_batch = model(measure_tensor)

                # 信号loss
                HW = simulate_signal_torch(params_batch, frequencies_sel)  # [BATCH_SIZE, N_freq]
                Fsim_obj = Fref_obj.clone().detach() * HW
                Fmeas_obj_tensor = Fmeas_obj.clone().detach()
                signal_loss = loss_func_torch(
                    Fsim_obj,
                    Fmeas_obj_tensor,
                    N, ml1, ml2, ml3, ml4,
                    plot_info={'batch_idx': batch_idx, 'epoch': epoch}
                )

                # 厚度loss
                pred_thickness = params_batch[:, 12:15]  # [BATCH_SIZE, 3]
                label_thickness = thickness_label_batch.clone().detach()
                thickness_loss = thickness_loss_fn(pred_thickness, label_thickness, config)

                # 自适应权重（每N步或每epoch更新一次）
                if batch_idx % 50 == 0 or batch_idx == 0:
                    def signal_loss_fn():
                        HW = simulate_signal_torch(model(measure_tensor), frequencies_sel)
                        Fsim_obj = Fref_obj.clone().detach() * HW
                        Fmeas_obj_tensor = Fmeas_obj.clone().detach()
                        return loss_func_torch(
                            Fsim_obj,
                            Fmeas_obj_tensor,
                            N, ml1, ml2, ml3, ml4,
                            plot_info=None
                        )
                    def thickness_loss_fn_local():
                        pred_thickness = model(measure_tensor)[:, 12:15]
                        return thickness_loss_fn(pred_thickness, label_thickness, config)

                    w_signal, w_thickness = compute_adaptive_weights(model, measure_tensor, label_thickness, signal_loss_fn, thickness_loss_fn_local)
                # 否则沿用上一次的w_signal, w_thickness
                total_loss = w_thickness * thickness_loss + w_signal * signal_loss
                optimizer.zero_grad()
                total_loss.backward()  # 普通float32反向传播
                optimizer.step()
                epoch_loss += total_loss.item()
                # 打印loss
                if (batch_idx+1) % 10 == 0:
                    print(f"[TRAIN] signal_loss={signal_loss.item():.7f}, thickness_loss={thickness_loss.item():.7f}, total_loss={thickness_loss.item() + signal_loss.item():.7f}")
                # TensorBoard记录
                global_step = epoch * len(dataloader) + batch_idx
                writer_train.add_scalar('Waveform', signal_loss.item(), global_step)
                writer_train.add_scalar('Actual_Thickness', thickness_loss.item() / config['lambda_supervise'], global_step)
                # 保留原有重要打印信息
                # ... existing code ...
                # ====== 每20个batch评估一次验证集 ======
                if (batch_idx+1) % 20 == 0:
                    val_data_loss, val_thickness_loss = evaluate_on_val(
                        model, val_dataloader, device, max_batches=config['max_val_batches']
                    )
                    print(f'[VAL] batch {batch_idx+1}: val_data_loss={val_data_loss:.2e}, val_thickness_loss={val_thickness_loss / config['lambda_supervise']:.2e}')
                    writer_val.add_scalar('Waveform', val_data_loss, global_step)
                    writer_val.add_scalar('Actual_Thickness', val_thickness_loss / config['lambda_supervise'], global_step)
                    # scheduler.step(val_data_loss)  # 根据验证集loss调整学习率
                    model.train()

        # 每个epoch结束后指数衰减学习率
        scheduler.step()

        # 每30个epoch保存一次模型参数
        if (epoch + 1) % 30 == 0:
            os.makedirs('./checkpoints', exist_ok=True)
            save_path = f'./checkpoints/0728_model_epoch{epoch+1}.pth'
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_path)
            print(f"模型已保存到 {save_path}")

        # ====== 打印每一层参数的梯度 ======
        print(f"[Epoch {epoch+1}] 每一层参数的梯度信息:")
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                print(f"  {name}: shape={tuple(param.shape)}, grad_norm={grad_norm:.4e}")
            else:
                print(f"  {name}: shape={tuple(param.shape)}, grad=None")

        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        print(f"[梯度] 当前神经网络所有参数的梯度二范数: {total_grad_norm:.4e}")

        # ====== 打印当前学习率 ======
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch+1}] 当前学习率: {current_lr:.6e}")
        # ====== 打印最后一个batch第一个样本的16个参数和label ======
        try:
            print("[Epoch {}] 最后一个batch第一个样本的16个预测参数: ".format(epoch+1), params_batch[0].detach().cpu().numpy())
            print("[Epoch {}] 最后一个batch第一个样本的label: ".format(epoch+1), thickness_label_batch[0].detach().cpu().numpy())
        except Exception as e:
            print(f"[Epoch {epoch+1}] 打印参数和label时出错: {e}")




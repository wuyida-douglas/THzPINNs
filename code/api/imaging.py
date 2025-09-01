#这个代码用于把成像数据送入模型，得到每一个点对应的厚度，然后输出到一个csv文件中
import os
import numpy as np
import torch
from models.MLP import SimpleMLP

import glob
import csv

# ====== 配置参数 ======
config = {
    'DATA_ROOT': r'./data/成像集/1225',  # 需要修改为你的数据文件夹
    'input_length': 3451,
    'checkpoint_path': './checkpoints/0728_model_epoch240.pth',  # 你的模型权重
    'scale': 1/(2.2e4),
    'output_csv': 'scanning.csv',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

def load_signal(file_path, scale):
    data = np.loadtxt(file_path)
    # 假设第一列是时间，第二列是信号
    signal = torch.from_numpy(data[:, 1] * scale).float()
    return signal

def main():
    device = torch.device(config['device'])
    print(f'使用设备: {device}')
    # 加载模型
    model = SimpleMLP(input_length=config['input_length'], dropout=0)
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # 遍历所有 *_1.txt 文件
    pattern = os.path.join(config['DATA_ROOT'], '*_1.txt')
    file_list = glob.glob(pattern)
    print(f'共找到 {len(file_list)} 个待成像文件')

    results = []
    for file_path in file_list:
        base = os.path.basename(file_path)
        try:
            parts = base.split('_')
            if len(parts) < 2:
                print(f"文件名解析失败: {base}, 跳过。")
                continue
            x = int(parts[0])
            y = int(parts[1])
        except Exception as e:
            print(f"文件名解析失败: {base}, 跳过。错误: {e}")
            continue
        # 读取信号和时间
        data = np.loadtxt(file_path)
        time = torch.from_numpy(data[:, 0]).float()
        signal = torch.from_numpy(data[:, 1] * config['scale']).float()
        if signal.shape[0] != config['input_length']:
            print(f"信号长度不符: {file_path}, 跳过")
            continue
        # ====== 预处理流程（与val一致） ======
        # 1. FFT
        ml1, ml2, ml3, ml4 = 2, 1725, 1726, 3449
        N = time.shape[0]
        dt_ps = (time[1] - time[0])  # 标量
        N0 = N
        device = torch.device(config['device'])
        frequencies_Hz = torch.fft.fftfreq(N0, d=dt_ps.item() * 1e-12, device=device)
        frequencies_THz = frequencies_Hz / 1e12
        # 2. FFT
        signal = signal.to(device)
        signal_fft = torch.fft.fft(signal)
        Fmeas_obj = signal_fft[ml1:ml2+1]
        # 3. 重组频谱
        measure_full_fft = torch.zeros_like(signal_fft, dtype=torch.complex128, device=device)
        measure_full_fft[ml1:ml2+1] = Fmeas_obj
        measure_full_fft[ml3:ml4+1] = torch.conj(torch.flip(Fmeas_obj, dims=[0]))
        # 4. IFFT
        measure_filtered = torch.fft.ifft(measure_full_fft).real.float()
        # 5. 送入模型
        signal_tensor = measure_filtered.unsqueeze(0).unsqueeze(0)  # [1, 1, L]
        with torch.no_grad():
            output = model(signal_tensor)
            thickness = output[0, 12:15].cpu().numpy()
        results.append([x, y, thickness[0], thickness[1], thickness[2]])

    # 保存为csv
    with open(config['output_csv'], 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'thickness1', 'thickness2', 'thickness3'])
        for row in results:
            writer.writerow(row)
    print(f'成像结果已保存到 {config['output_csv']}')

if __name__ == '__main__':
    main() 
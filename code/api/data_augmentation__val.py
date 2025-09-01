import scipy.io as sio
import numpy as np
import os
import random
from typing import Optional


def load_signal_from_mat(mat_path, var_name='reflected_total'):
    mat_data = sio.loadmat(mat_path)
    if var_name not in mat_data:
        raise ValueError(f"MAT文件中未找到'{var_name}'变量: {mat_path}")
    return mat_data[var_name].flatten()


def augment_signal(signal, amp=1.0, noise_range=(0, 0), crop_start: Optional[int]=None, crop_len: Optional[int]=None):
    # 1. 幅度缩放
    signal = signal * amp
    # 2. 滑动窗口裁切
    if crop_start is not None and crop_len is not None:
        signal = signal[crop_start:crop_start+crop_len]
    # 3. 加噪声
    noise = np.random.uniform(*noise_range, size=signal.shape)
    signal = signal + noise
    return signal


def save_txt(time, signal, out_path):
    data_matrix = np.column_stack((time, signal))
    np.savetxt(out_path, data_matrix, fmt='%.10f %.10f')


def batch_augment_measure_and_ref(
    measure_root, ref_root, output_root, augment_times=1, amp_range=(1.30,1.30), noise_range=(0,0), var_name='reflected_total', crop_len=3451, crop_start_range=(0, 1000)):
    # 获取所有参考信号mat路径
    ref_files = [os.path.join(ref_root, f) for f in os.listdir(ref_root) if f.lower().endswith('.mat')]
    if not ref_files:
        raise RuntimeError('参考信号文件夹为空！')
    for subdir, dirs, files in os.walk(measure_root):
        rel_subdir = os.path.relpath(subdir, measure_root)
        for file in files:
            if not file.lower().endswith('.mat'):
                continue
            measure_mat_path = os.path.join(subdir, file)
            measure_signal = load_signal_from_mat(measure_mat_path, var_name)
            # 生成时间轴
            time_axis = np.linspace(0, 100, len(measure_signal))
            for aug_idx in range(1, augment_times+1):
                # 随机选一个参考信号
                ref_mat_path = random.choice(ref_files)
                ref_signal = load_signal_from_mat(ref_mat_path, var_name)
                # 随机增强参数
                amp = np.random.uniform(*amp_range)
                crop_start = 150 #2
                # 增强（同参数）
                try:
                    measure_aug = augment_signal(measure_signal, amp=amp, noise_range=noise_range, crop_start=crop_start, crop_len=crop_len)
                    ref_aug = augment_signal(ref_signal, amp=amp, noise_range=noise_range, crop_start=crop_start, crop_len=crop_len)
                except Exception as e:
                    print(f"增强失败: {measure_mat_path}, {ref_mat_path}, 错误: {e}")
                    continue
                time_crop = time_axis[crop_start:crop_start+crop_len]
                # 构造输出路径
                base_name = os.path.splitext(file)[0]
                out_subdir = os.path.join(output_root, rel_subdir)
                os.makedirs(out_subdir, exist_ok=True)
                measure_out = os.path.join(out_subdir, f"{base_name}_{aug_idx}_1.txt")
                ref_out = os.path.join(out_subdir, f"{base_name}_{aug_idx}_2.txt")
                save_txt(time_crop, measure_aug, measure_out)
                save_txt(time_crop, ref_aug, ref_out)
                print(f"保存: {measure_out}, {ref_out}")


if __name__ == "__main__":
    # 主文件夹（实测信号）
    measure_root = r'./data/验证集'
    # 参考信号文件夹
    ref_root = r'./data/参考信号'
    # 输出文件夹
    output_root = r'./data/验证集_0728'
    batch_augment_measure_and_ref(
        measure_root, ref_root, output_root,#amp_range=(1.35,1.35)
        augment_times=3, amp_range=(1.00,1.00), noise_range=(0,0), crop_len=3451, var_name='reflected_total'
    )
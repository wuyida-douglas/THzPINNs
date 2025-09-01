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


def augment_signal(signal, amp=1.0, noise_range=(0, 5), crop_start: Optional[int]=None, crop_len: Optional[int]=None):
    # 1. 幅度缩放
    signal = signal * amp
    # 2. 滑动窗口裁切
    if crop_start is not None and crop_len is not None:
        print(f"[augment_signal] 裁剪前长度: {len(signal)}, crop_start: {crop_start}, crop_len: {crop_len}")
        signal = signal[crop_start:crop_start+crop_len]
        print(f"[augment_signal] 裁剪后长度: {len(signal)}")
    # 3. 加噪声
    noise = np.random.uniform(*noise_range, size=signal.shape)
    signal = signal + noise
    return signal


def save_txt(time, signal, out_path):
    data_matrix = np.column_stack((time, signal))
    np.savetxt(out_path, data_matrix, fmt='%.10f %.10f')


def mixup_signals(sig1, sig2, lam=None):
    if lam is None:
        lam = np.random.uniform(0, 1)
    return lam * sig1 + (1 - lam) * sig2


def batch_mixup_augment(
    measure_root, ref_root, output_root, augment_times=2, amp_ranges=[(0.9,1.1),], noise_range=(-4,4), var_name='reflected_total', crop_len=3051):
    # 获取所有参考信号mat路径
    ref_files = [os.path.join(ref_root, f) for f in os.listdir(ref_root) if f.lower().endswith('.mat')]
    if not ref_files:
        raise RuntimeError('参考信号文件夹为空！')
    # 遍历每个子文件夹
    for subdir, dirs, files in os.walk(measure_root):
        rel_subdir = os.path.relpath(subdir, measure_root)
        mat_files = [f for f in files if f.lower().endswith('.mat')]
        if len(mat_files) < 2:
            print("出问题啦")
            continue  # 至少需要2个信号做mixup
        for aug_idx in range(1, augment_times+1):
            # 随机选2个实测信号
            m1, m2 = random.sample(mat_files, 2)
            m1_path = os.path.join(subdir, m1)
            m2_path = os.path.join(subdir, m2)
            sig1 = load_signal_from_mat(m1_path, var_name)
            sig2 = load_signal_from_mat(m2_path, var_name)
            # mixup
            lam1 = np.random.uniform(0, 1)
            measure_mix = mixup_signals(sig1, sig2, lam1)
            print(f"[batch_mixup_augment] mixup后信号长度: {len(measure_mix)}")
            # 生成时间轴
            time_axis = np.linspace(0, 100, len(measure_mix))
            # 随机选2个参考信号
            r1_path, r2_path = random.sample(ref_files, 2)
            ref1 = load_signal_from_mat(r1_path, var_name)
            ref2 = load_signal_from_mat(r2_path, var_name)
            lam2 = np.random.uniform(0, 1)
            ref_mix = mixup_signals(ref1, ref2, lam2)
            # 随机选amp区间
            amp_range = random.choice(amp_ranges)
            amp = np.random.uniform(*amp_range)
            # 随机滑动窗口
            crop_start = random.randint(0, 300)#(112, 600)
            # 增强（同参数）
            try:
                measure_aug = augment_signal(measure_mix, amp=amp, noise_range=noise_range, crop_start=crop_start, crop_len=crop_len)
                ref_aug = augment_signal(ref_mix, amp=amp, noise_range=noise_range, crop_start=crop_start, crop_len=crop_len)
            except Exception as e:
                print(f"增强失败: {m1_path}, {m2_path}, {r1_path}, {r2_path}, 错误: {e}")
                continue
            time_crop = time_axis[crop_start:crop_start+crop_len]
            # 构造输出路径

            print(len(measure_aug))
            print(len(ref_aug))

            out_subdir = os.path.join(output_root, rel_subdir)
            os.makedirs(out_subdir, exist_ok=True)
            base_name = f"mixup_{aug_idx}"
            measure_out = os.path.join(out_subdir, f"{base_name}_1.txt")
            ref_out = os.path.join(out_subdir, f"{base_name}_2.txt")
            save_txt(time_crop, measure_aug, measure_out)
            save_txt(time_crop, ref_aug, ref_out)
            print(f"保存: {measure_out}, {ref_out}")


if __name__ == "__main__":
    # 主文件夹（实测信号）
    measure_root = r'./data/训练集'
    # 参考信号文件夹
    ref_root = r'./data/参考信号'
    # 输出文件夹
    output_root = r'./data/训练集_0728'

    batch_mixup_augment(
        measure_root, ref_root, output_root,
        augment_times=1800, amp_ranges=[(0.50,1.50)], noise_range=(-4,4), var_name='reflected_total', crop_len=3451
    )
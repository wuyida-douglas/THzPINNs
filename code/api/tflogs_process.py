#这个代码用于将服务器保存的tensorboard文件，变成方便读取的mat文件
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import scipy.io
import numpy as np

def print_all_scalar_tags(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    print(f"Scalar tags in {log_dir}:")
    print(ea.Tags().get('scalars', []))

# 修改为你的实际日志文件夹路径
train_log_dir = r'D:\实验图片\test\没有PINNs的结果\train'  # 例如 './runs/train'
val_log_dir = r'D:\实验图片\test\没有PINNs的结果\val'      # 例如 './runs/val'

print_all_scalar_tags(train_log_dir)
print_all_scalar_tags(val_log_dir)

def extract_scalars(log_dir, tag):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    if tag not in ea.Tags()['scalars']:
        print(f"Tag '{tag}' not found in {log_dir}")
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

# 提取并保存数据到mat文件（每个mat文件包含train和val的x、y）
# Waveform
train_waveform_steps, train_waveform_values = extract_scalars(train_log_dir, 'Waveform')
val_waveform_steps, val_waveform_values = extract_scalars(val_log_dir, 'Waveform')
waveform_mat_path = os.path.join(os.path.dirname(train_log_dir), 'Waveform.mat')
scipy.io.savemat(waveform_mat_path, {
    'train_x': np.array(train_waveform_steps).reshape(-1, 1),
    'train_y': np.array(train_waveform_values).reshape(-1, 1),
    'val_x': np.array(val_waveform_steps).reshape(-1, 1),
    'val_y': np.array(val_waveform_values).reshape(-1, 1)
})

# Actual_Thickness
train_thickness_steps, train_thickness_values = extract_scalars(train_log_dir, 'Actual_Thickness')
val_thickness_steps, val_thickness_values = extract_scalars(val_log_dir, 'Actual_Thickness')
# 对y轴先除以3.2再开方
train_thickness_y_processed = np.sqrt(np.array(train_thickness_values) / 3.2).reshape(-1, 1)
val_thickness_y_processed = np.sqrt(np.array(val_thickness_values) / 3.2).reshape(-1, 1)
thickness_mat_path = os.path.join(os.path.dirname(train_log_dir), 'Actual_Thickness.mat')
scipy.io.savemat(thickness_mat_path, {
    'train_x': np.array(train_thickness_steps).reshape(-1, 1),
    'train_y': train_thickness_y_processed,
    'val_x': np.array(val_thickness_steps).reshape(-1, 1),
    'val_y': val_thickness_y_processed
})

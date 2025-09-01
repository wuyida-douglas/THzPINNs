#这个代码用来用提取出的材料参数绘制介电常数曲线
import pandas as pd
import numpy as np
from physics_torch import lorentz_eps_torch_batch
import torch

def plot_dielectric_constant():
    # 读取xlsx文件
    df = pd.read_excel(r'C:\Users\wuyida\Desktop\0728最后一搏\parameters_prediction.xlsx')
    print(f"读取到 {len(df)} 个样本的参数")
    
    # 定义频率范围 (与训练时一致)
    ml1, ml2 = 2, 1725
    N = 3451
    # 使用实际的采样间隔计算
    dt_ps = 100 / 3751  # 时间间隔 = 100ps / 3751
    frequencies_Hz = np.fft.fftfreq(N, d=dt_ps * 1e-12)
    frequencies_THz = frequencies_Hz / 1e12
    frequencies_sel = frequencies_THz[ml1:ml2+1]  # 选择特定频率范围
    
    print(f"采样间隔: {dt_ps:.6f} ps")
    print(f"频率范围: {frequencies_sel[0]:.3f} - {frequencies_sel[-1]:.3f} THz")
    print(f"频率点数: {len(frequencies_sel)}")
    
    # 转换为torch tensor
    freq_tensor = torch.tensor(frequencies_sel, dtype=torch.float64)
    
    # 准备数据存储
    all_data = {}
    all_data['频率_THz'] = frequencies_sel
    
    # 为每个样本计算三层油漆的介电常数
    for sample_idx in range(len(df)):
        print(f"正在处理第 {sample_idx + 1} 个样本...")
        
        # 获取当前样本的前12个参数
        params = df.iloc[sample_idx].values[:12]
        
        # 转换为torch tensor
        params_tensor = torch.tensor(params, dtype=torch.float64).unsqueeze(0)  # [1, 12]
        
        # 提取每层的参数
        # 第一层：参数0-3
        einf1, wp1, w01, gamma1 = params_tensor[0, 0], params_tensor[0, 1], params_tensor[0, 2], params_tensor[0, 3]
        # 第二层：参数4-7
        einf2, wp2, w02, gamma2 = params_tensor[0, 4], params_tensor[0, 5], params_tensor[0, 6], params_tensor[0, 7]
        # 第三层：参数8-11
        einf3, wp3, w03, gamma3 = params_tensor[0, 8], params_tensor[0, 9], params_tensor[0, 10], params_tensor[0, 11]
        
        # 计算每层的介电常数
        eps1 = lorentz_eps_torch_batch(einf1.unsqueeze(0), wp1.unsqueeze(0), w01.unsqueeze(0), gamma1.unsqueeze(0), freq_tensor.unsqueeze(0))
        eps2 = lorentz_eps_torch_batch(einf2.unsqueeze(0), wp2.unsqueeze(0), w02.unsqueeze(0), gamma2.unsqueeze(0), freq_tensor.unsqueeze(0))
        eps3 = lorentz_eps_torch_batch(einf3.unsqueeze(0), wp3.unsqueeze(0), w03.unsqueeze(0), gamma3.unsqueeze(0), freq_tensor.unsqueeze(0))
        
        # 转换为numpy数组
        eps1_np = eps1[0].numpy()
        eps2_np = eps2[0].numpy()
        eps3_np = eps3[0].numpy()
        
        # 存储数据
        all_data[f'样本{sample_idx + 1}_第1层_实部'] = np.real(eps1_np)
        all_data[f'样本{sample_idx + 1}_第1层_虚部'] = np.imag(eps1_np)
        all_data[f'样本{sample_idx + 1}_第2层_实部'] = np.real(eps2_np)
        all_data[f'样本{sample_idx + 1}_第2层_虚部'] = np.imag(eps2_np)
        all_data[f'样本{sample_idx + 1}_第3层_实部'] = np.real(eps3_np)
        all_data[f'样本{sample_idx + 1}_第3层_虚部'] = np.imag(eps3_np)
    
    # 创建DataFrame并保存到xlsx
    result_df = pd.DataFrame(all_data)
    result_df.to_excel(r'C:\Users\wuyida\Desktop\0728最后一搏\dielectric_constant_data.xlsx', index=False)
    print(f"数据已保存到 dielectric_constant_data.xlsx")
    print(f"共保存了 {len(result_df)} 行数据，{len(result_df.columns)} 列")
    print(f"列标题：{list(result_df.columns)}")

if __name__ == '__main__':
    plot_dielectric_constant() 
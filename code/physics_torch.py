import torch

def simulate_signal_torch(params, freq_sel):
    # 支持params: [B, 16] 或 [16]
    # freq_sel: [N]
    is_batched = params.ndim == 2
    if not is_batched:
        params = params.unsqueeze(0)  # [1, 16]
    B = params.shape[0]
    N = freq_sel.shape[-1]
    freq_sel_t = freq_sel.to(torch.complex128)
    if freq_sel_t.ndim == 1:
        freq_sel_t = freq_sel_t.unsqueeze(0).expand(B, -1)  # [B, N]
    # 拆分参数
    einf1, wp1, w01, gamma1 = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
    einf2, wp2, w02, gamma2 = params[:, 4], params[:, 5], params[:, 6], params[:, 7]
    einf3, wp3, w03, gamma3 = params[:, 8], params[:, 9], params[:, 10], params[:, 11]
    d1 = params[:, 12] * 1e-6
    d2 = params[:, 13] * 1e-6
    d3 = params[:, 14] * 1e-6
    d_comp = params[:, 15] * 1e-6
    # 计算每层介电常数（批量）
    eps1 = lorentz_eps_torch_batch(einf1, wp1, w01, gamma1, freq_sel_t)
    eps2 = lorentz_eps_torch_batch(einf2, wp2, w02, gamma2, freq_sel_t)
    eps3 = lorentz_eps_torch_batch(einf3, wp3, w03, gamma3, freq_sel_t)
    # 三层rouard反射率（批量）
    sim_reflect = rouard_reflectance_torch_batch(freq_sel_t, eps1, eps2, eps3, d1, d2, d3)
    phase_comp = torch.exp(2j * torch.pi * freq_sel_t * 1e12 * d_comp.unsqueeze(1) * 2 / 3e8)
    sim_reflect = sim_reflect * phase_comp
    # 加断言，确保没有NaN或Inf
    assert not torch.isnan(sim_reflect).any(), "[simulate_signal_torch] sim_reflect contains NaN"
    assert not torch.isinf(sim_reflect).any(), "[simulate_signal_torch] sim_reflect contains Inf"
    if not is_batched:
        return sim_reflect[0]
    return sim_reflect

def lorentz_eps_torch_batch(einf, wp, w0, gamma, freq_t):
    # 所有输入: [B], freq_t: [B, N]
    omega = 2 * torch.pi * freq_t
    einf = einf.unsqueeze(1)
    wp = wp.unsqueeze(1)
    w0 = w0.unsqueeze(1)
    gamma = gamma.unsqueeze(1)
    # 分母加epsilon，防止为0
    denom = (w0 ** 2 - omega ** 2 - 1j * gamma * omega)
    denom = denom + 1e-8 * (denom.abs() < 1e-8)
    eps = einf + (wp ** 2) / denom
    return eps  # [B, N]

def eps_to_n_torch(eps):
    # sqrt输入加clamp，防止负数
    real_eps = torch.real(eps)
    abs_eps = torch.abs(eps)
    rn = torch.sqrt((abs_eps + real_eps) / 2)
    ik = torch.sqrt((abs_eps - real_eps) / 2)
    return rn - 1j * ik

def rouard_reflectance_torch_batch(frequencies, eps1, eps2, eps3, d1, d2, d3, angle=20, c=3e8):
    # frequencies: [B, N], eps1/2/3: [B, N], d1/2/3: [B]
    B, N = frequencies.shape
    device = frequencies.device
    dtype = frequencies.dtype
    theta1 = torch.deg2rad(torch.tensor(angle, dtype=torch.float64, device=device))
    n1 = torch.tensor(1.0, dtype=dtype, device=device)
    n2 = eps_to_n_torch(eps1)
    n3 = eps_to_n_torch(eps2)
    n4 = eps_to_n_torch(eps3)
    # n5 = 1j * 1e10  # 基底
    theta1 = theta1.to(dtype)
    # 逐层计算入射角，直接用复数链路
    sin_theta2 = (n1 / n2) * torch.sin(theta1)
    theta2 = torch.arcsin(sin_theta2)
    sin_theta3 = (n1 / n3) * torch.sin(theta1)
    theta3 = torch.arcsin(sin_theta3)
    sin_theta4 = (n1 / n4) * torch.sin(theta1)
    theta4 = torch.arcsin(sin_theta4)
    def r_TE(n_i, n_j, theta_i, theta_j):
        denom = n_i * torch.cos(theta_i) + n_j * torch.cos(theta_j)
        denom = denom + 1e-8 * (denom.abs() < 1e-8)
        return -(n_i * torch.cos(theta_i) - n_j * torch.cos(theta_j)) / denom
    def t_TE(n_i, n_j, theta_i, theta_j):
        denom = n_i * torch.cos(theta_i) + n_j * torch.cos(theta_j)
        denom = denom + 1e-8 * (denom.abs() < 1e-8)
        return (2 * n_i * torch.cos(theta_i)) / denom
    # 反射和透射系数
    r12 = r_TE(n1, n2, theta1, theta2)
    r23 = r_TE(n2, n3, theta2, theta3)
    r34 = r_TE(n3, n4, theta3, theta4)
    r45 = 1
    t12 = t_TE(n1, n2, theta1, theta2)
    t21 = t_TE(n2, n1, theta2, theta1)
    t23 = t_TE(n2, n3, theta2, theta3)
    t32 = t_TE(n3, n2, theta3, theta2)
    t34 = t_TE(n3, n4, theta3, theta4)
    t43 = t_TE(n4, n3, theta4, theta3)
    # 相移，分母加epsilon
    Beta2 = ((2 * torch.pi * frequencies * 1e12) / c) * n2 * d1.unsqueeze(1) / (torch.cos(theta2) + 1e-8)
    Beta3 = ((2 * torch.pi * frequencies * 1e12) / c) * n3 * d2.unsqueeze(1) / (torch.cos(theta3) + 1e-8)
    Beta4 = ((2 * torch.pi * frequencies * 1e12) / c) * n4 * d3.unsqueeze(1) / (torch.cos(theta4) + 1e-8)
    # 递推公式（3层），分母加epsilon
    Hw4 = r45
    def safe_div(num, denom):
        denom = denom + 1e-8 * (denom.abs() < 1e-8)
        return num / denom
    Hw3 = r34 + safe_div(t34 * t43 * Hw4 * torch.exp(-2j * Beta4), (1 + r34 * Hw4 * torch.exp(-2j * Beta4)))
    Hw2 = r23 + safe_div(t23 * t32 * Hw3 * torch.exp(-2j * Beta3), (1 + r23 * Hw3 * torch.exp(-2j * Beta3)))
    Hw1 = r12 + safe_div(t12 * t21 * Hw2 * torch.exp(-2j * Beta2), (1 + r12 * Hw2 * torch.exp(-2j * Beta2)))
    return Hw1  # [B, N] 
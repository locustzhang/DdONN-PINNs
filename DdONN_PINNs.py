#DdONN-PINNs: Complex Optical Wavefield Reconstruction via Domain-Decomposed Optical Neural Networks with Physics-Informed Constraints
#This repository contains the implementation of DdONN-PINNs, a hybrid algorithm designed for high-precision reconstruction of complex optical wavefields. The core of the algorithm lies in integrating domain decomposition strategies with optical neural networks (ONNs) and physics-informed neural networks (PINNs), enabling it to adapt to spatial heterogeneity of wavefields (e.g., high-gradient cores vs. smooth edges) while preserving physical consistency.

#For technical details, refer to the associated research paper. Questions or feedback are welcome via the contact below.

#Author: Lipu Zhang
#Contact: zhanglipu@cuz.edu.cn
#%DomainONN-PINNs算法(Sin函数激活)
import numpy as np                  # 用于数值计算和数组操作
import torch                         # PyTorch深度学习框架核心
import torch.nn as nn                # 神经网络模块，用于构建模型
import os                            # 用于文件和目录操作
import matplotlib.pyplot as plt       # 用于数据可视化
import warnings                      # 新增：用于控制警告输出
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # 用于局部放大图
from torch.optim import AdamW         # 带权重衰减的优化器，有助于防止过拟合
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率调度器
from torch.autograd.functional import jacobian, hessian  # 用于计算雅各比矩阵和海森矩阵（导数）
from tqdm import trange              # 用于显示训练进度条

# ---------- 新增：过滤特定警告 ----------
# 屏蔽"tight_layout不兼容"的UserWarning
warnings.filterwarnings(
    "ignore",
    message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.",
    category=UserWarning
)

# ---------- 期刊级图形配置 ----------
plt.rcParams.update({
    "font.family": "Times New Roman",  # 期刊标准字体
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "axes.grid": False,  # 期刊图表通常无网格
    "savefig.dpi": 1000,  # 高分辨率，满足印刷要求
    "savefig.format": "pdf",  # 矢量格式，支持无损缩放
    "savefig.bbox": "tight",  # 去除多余空白
    "figure.figsize": (8, 6)  # 标准图表尺寸
})


# ---------- 自定义组件：将函数转换为PyTorch模块 ----------
class Lambda(nn.Module):
    """
    自定义Lambda层，用于在神经网络序列(nn.Sequential)中插入自定义函数
    这里主要用于包装正弦激活函数，适配SIREN网络架构
    """

    def __init__(self, func):
        super().__init__()
        self.func = func  # 存储传入的函数（如torch.sin）

    def forward(self, x):
        """前向传播：直接调用存储的函数处理输入"""
        return self.func(x)


# ---------- 环境配置与超参数设置 ----------
# 设置随机种子，确保实验结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 自动选择计算设备（优先使用GPU，无GPU则使用CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")  # 打印当前使用的设备

# 核心超参数配置（保持不变）
N, epochs = 64, 3000  # N: 网格尺寸（N×N）；epochs: 训练轮次
lr = 3e-4  # 初始学习率
hidden_dim = 512  # 神经网络隐藏层维度（控制模型表达能力）
noise_std = 0.05  # 噪声标准差，用于生成带噪声的输入数据
warmup_epochs = 100  # 热身轮次：前100轮主要优化数据损失，物理约束权重为0
weight_ramp_epochs = 500  # 权重渐变轮次：物理约束权重从0逐渐增加到1
lambda_phase_smooth = 0.1  # 相位平滑损失的权重系数

# 创建结果保存目录（若不存在则自动创建）
save_dir = "english_results_optimized_wave"
os.makedirs(save_dir, exist_ok=True)

# ---------- 数据生成与初始化 ----------
# 生成坐标网格：x范围[-5,5]，y范围[-3,3]，各包含N个采样点
x = torch.linspace(-5, 5, N, device=device, requires_grad=False)  # x轴坐标
y = torch.linspace(-3, 3, N, device=device, requires_grad=False)  # y轴坐标
X, Y = torch.meshgrid(x, y, indexing='ij')  # 生成2D网格坐标，shape为[N, N]

# 定义分域掩码：区分中心波峰区域和边缘区域（分域训练的核心）
# 中心区域覆盖波峰密集区域（X±3，Y±1.5），边缘区域为剩余部分
center_mask = (torch.abs(X) < 3) & (torch.abs(Y) < 1.5)  # 中心区域掩码（True表示中心）
edge_mask = ~center_mask  # 边缘区域掩码（与中心区域互补）

# 计算网格间距（用于导数计算和傅里叶变换）
dx = (x[-1] - x[0]).item() / (N - 1)  # x方向的网格步长
X_global = X.detach()  # 分离梯度，作为全局x坐标用于导数计算
Y_global = Y.detach()  # 分离梯度，作为全局y坐标用于导数计算


def rogue_wave(x, y):
    """
    生成真实的"怪波"(rogue wave)复振幅场，作为重建目标
    怪波是一种具有陡峭波峰和复杂干涉结构的特殊波动，常用于验证波场重建算法
    """
    # 计算分母（避免为0）
    denom = 1 + 4 * (x - y) ** 2 + 4 * y ** 2
    denom = torch.clamp(denom, min=1e-6)  # 限制最小值，防止除零错误
    # 计算分子（复数形式）
    numerator = 1 - 4 * (1 + 4j * (x - y))
    # 计算振幅（模长）和相位（指数形式）
    amplitude = torch.abs(numerator / denom)
    phase = torch.exp(1j * y)
    return amplitude * phase  # 复振幅 = 振幅 × 相位


# 生成波场数据
U_true = rogue_wave(X, Y).to(device)  # 真实无噪波场（复数，shape: [N, N]）
# 生成带噪声的波场（模拟实际测量数据）
U_noisy = U_true + noise_std * (torch.randn_like(U_true) + 1j * torch.randn_like(U_true)).to(device)
# 准备网络输入坐标（shape: [1, N, N, 2]，包含x和y坐标）
coords = torch.stack([X, Y], dim=-1).unsqueeze(0).to(device)
targets = U_true.unsqueeze(0).to(device)  # 重建目标（添加批次维度，shape: [1, N, N]）
true_amp_mean = torch.mean(torch.abs(U_true)).item()  # 真实振幅的均值（用于尺度约束）


# ---------- 网络架构：分域波场重建网络 ----------
class FourierDiffraction(nn.Module):
    """
    傅里叶域衍射层：模拟波场在自由空间中的传播过程
    基于傅里叶变换的快速衍射计算，比时域传播更高效
    """

    def __init__(self, N, dz, dx=dx):
        super().__init__()
        # 计算频率坐标（傅里叶域的x/y频率）
        fx = torch.fft.fftfreq(N, d=dx, device=device)  # x方向频率（单位：1/长度）
        FX, FY = torch.meshgrid(fx, fx, indexing='ij')  # 2D频率网格（shape: [N, N]）

        # 注册为非训练参数（buffer不参与梯度更新）
        self.register_buffer('FX', FX)  # x方向频率网格
        self.register_buffer('FY', FY)  # y方向频率网格

        self.dz = dz  # 传播距离（固定值）
        self.lam = nn.Parameter(torch.tensor(1.0, device=device))  # 波长（可学习参数）
        self.phase = nn.Parameter(torch.zeros(N, N, device=device))  # 额外相位调制（可学习）

    def forward(self, u):
        """
        前向传播：波场通过傅里叶域传播
        步骤：傅里叶变换 → 乘以传递函数 → 逆傅里叶变换
        """
        # 计算传递函数H（基于菲涅尔衍射公式）
        H = torch.exp(-1j * np.pi * self.lam * self.dz * (self.FX ** 2 + self.FY ** 2))
        # 傅里叶变换后乘以传递函数和相位调制，再逆变换得到传播后的波场
        return torch.fft.ifft2(torch.fft.fft2(u) * H * torch.exp(1j * self.phase))


class DomainONN(nn.Module):
    """
    分域光学神经网络（Domain-specific Optical Neural Network）：
    基于SIREN（正弦激活网络）架构，结合分域训练和复数处理，专门用于波场重建
    """

    def __init__(self, hidden_dim=512, N=64, dx=dx):
        super().__init__()
        # 共享特征提取网络（SIREN架构）：
        # SIREN使用正弦激活，适合建模连续波动信号（天然匹配波场的周期性）
        self.shared_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim), Lambda(lambda x: torch.sin(x)),  # 输入层→隐藏层+正弦激活
            nn.Linear(hidden_dim, hidden_dim), Lambda(lambda x: torch.sin(x)),  # 隐藏层1
            nn.Linear(hidden_dim, hidden_dim), Lambda(lambda x: torch.sin(x)),  # 隐藏层2
        )

        # 分域输出头：针对中心区和边缘区的不同特性单独优化
        self.center_head = nn.Sequential(  # 中心区输出头（波峰密集，结构复杂）
            nn.Linear(hidden_dim, hidden_dim), Lambda(lambda x: torch.sin(x)),
            nn.Linear(hidden_dim, 2)  # 输出2维：实部+虚部
        )
        self.edge_head = nn.Sequential(  # 边缘区输出头（结构简单，变化平缓）
            nn.Linear(hidden_dim, hidden_dim), Lambda(lambda x: torch.sin(x)),
            nn.Linear(hidden_dim, 2)  # 输出2维：实部+虚部
        )

        # SIREN专用初始化：正弦激活需特殊权重初始化（避免梯度消失/爆炸）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.in_features  # 输入特征数
                # 权重初始化范围：[-√(6/fan_in), √(6/fan_in)]（SIREN论文推荐）
                nn.init.uniform_(m.weight, -np.sqrt(6 / fan_in), np.sqrt(6 / fan_in))
                nn.init.zeros_(m.bias)  # 偏置初始化为0

        # 复数卷积层：对生成的波场进行空间平滑（去除高频噪声）
        self.conv = nn.Conv2d(
            1, 1, kernel_size=3, padding=1, bias=True,
            dtype=torch.cfloat  # 显式指定复数类型（匹配波场的复数特性）
        )
        # 复数卷积初始化：实部用线性激活的Kaiming初始化，虚部初始为0
        nn.init.kaiming_uniform_(self.conv.weight.real, nonlinearity='linear')
        nn.init.zeros_(self.conv.weight.imag)  # 虚部初始为0（减少初始复杂度）
        nn.init.zeros_(self.conv.bias)  # 偏置初始为0

        # 多尺度衍射传播：串联两个不同传播距离的衍射层（覆盖更多传播模式）
        self.diffraction = nn.Sequential(
            FourierDiffraction(N, 0.5),  # 短距离传播（dz=0.5）
            FourierDiffraction(N, 1.5)  # 长距离传播（dz=1.5）
        )

    def forward(self, coords):
        """前向传播：从坐标→特征提取→分域输出→复数波场→传播"""
        batch_size, h, w, _ = coords.shape  # coords shape: [1, N, N, 2]
        coords_flat = coords.view(-1, 2)  # 展平坐标为[N², 2]（便于MLP处理）

        # 共享特征提取：所有区域共享底层特征（如基础波动模式）
        feat = self.shared_mlp(coords_flat)  # 提取特征：[N², hidden_dim]

        # 分域输出：中心区和边缘区使用不同的输出头（针对区域特性优化）
        center_out = self.center_head(feat)  # 中心区输出：[N², 2]（实部+虚部）
        edge_out = self.edge_head(feat)  # 边缘区输出：[N², 2]（实部+虚部）

        # 合并分域结果：用掩码选择每个位置属于中心区还是边缘区的输出
        mask_flat = center_mask.flatten().to(device)  # 展平掩码：[N²]
        out = torch.where(
            mask_flat.unsqueeze(1),  # 扩展为[N², 1]，与输出维度匹配
            center_out,  # 掩码为True时选择中心区输出
            edge_out  # 掩码为False时选择边缘区输出
        )  # 合并后形状：[N², 2]

        # 组合为复数波场并进行后处理
        real = out[:, 0].view(batch_size, h, w)  # 实部：[1, N, N]
        imag = out[:, 1].view(batch_size, h, w)  # 虚部：[1, N, N]
        u0 = torch.complex(real, imag).unsqueeze(0)  # 组合为复数：[1, 1, N, N]
        u0 = self.conv(u0).squeeze(1)  # 复数卷积平滑（去除噪声），shape: [1, N, N]
        return self.diffraction(u0)  # 经过衍射传播后的输出波场


# 初始化模型并移动到计算设备
model = DomainONN(hidden_dim, N, dx=dx).to(device)


# ---------- 损失函数定义 ----------
def nlse_residual(u):
    """
    计算NLSE（非线性薛定谔方程）的物理残差及梯度（gPINN约束）
    残差越小，说明模型输出越符合物理规律
    """
    u = u.squeeze(0).requires_grad_(True)  # 移除批次维度并启用梯度计算：[N, N]
    u_r, u_i = u.real, u.imag  # 分离复数的实部和虚部

    # 计算一阶/二阶导数（使用自动微分）
    # 对y的一阶导数（实部和虚部）
    u_y_r = jacobian(lambda y: u_r.sum(), Y_global, create_graph=True)[0]
    u_y_i = jacobian(lambda y: u_i.sum(), Y_global, create_graph=True)[0]
    # 对x的二阶导数（实部和虚部）
    u_xx_r = hessian(lambda x: u_r.sum(), X_global, create_graph=True)[0, 0]
    u_xx_i = hessian(lambda x: u_i.sum(), X_global, create_graph=True)[0, 0]

    # 计算NLSE方程残差（复数形式）
    # NLSE方程：i∂u/∂y + 0.5∂²u/∂x² + |u|²u = 0 → 残差=上述表达式
    residual = (1j * torch.complex(u_y_r, u_y_i) +
                0.5 * torch.complex(u_xx_r, u_xx_i) +
                (torch.abs(u) ** 2) * u)

    # 计算残差的空间梯度（gPINN增强约束，确保残差空间平滑）
    res_grad_x = jacobian(lambda x: residual.real.sum(), X_global, create_graph=True)[0]
    res_grad_y = jacobian(lambda y: residual.real.sum(), Y_global, create_graph=True)[0]
    return residual, res_grad_x, res_grad_y


def aspinn_weight(residual):
    """
    ASPINN（自适应采样PINN）权重：对高残差区域赋予更高权重
    让模型更关注难以拟合的区域（如波峰边缘）
    """
    res_amp = torch.abs(residual)  # 残差的振幅
    return res_amp / (torch.mean(res_amp) + 1e-8)  # 归一化权重（避免数值过大）


def adaptive_weights(loss_data, loss_phys, loss_scale):
    """
    自适应权重：动态平衡不同损失项的贡献
    损失值越大的项，权重越低（避免某一损失主导优化）
    """
    w_data = 1.0 / (loss_data.detach() + 1e-8)  # 数据损失权重（detach避免影响梯度）
    w_phys = 1.0 / (loss_phys.detach() + 1e-8)  # 物理损失权重
    w_scale = 1.0 / (loss_scale.detach() + 1e-8)  # 尺度损失权重
    total_w = w_data + w_phys + w_scale  # 总权重
    return w_data / total_w, w_phys / total_w  # 归一化权重


# ---------- 训练流程 ----------
# 初始化优化器（带权重衰减，防止过拟合）
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
# 学习率调度器：余弦退火（逐渐降低学习率，提升收敛稳定性）
scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-7)

# 损失日志：记录训练过程中的各项损失
loss_log = {
    'total': [], 'data': [], 'phys': [],
    'gpinna': [], 'scale': [], 'phase_smooth': []
}

# 初始预测评估（训练前查看模型初始输出）
with torch.no_grad():  # 禁用梯度计算，加快速度
    u_init = model(coords)
    print(f"Initial amp mean: {torch.mean(torch.abs(u_init)).item():.4f} "
          f"(Target: {true_amp_mean:.4f})")

# 训练循环（带进度条）
pbar = trange(1, epochs + 1, ncols=100)
for epoch in pbar:
    model.train()  # 切换到训练模式
    optimizer.zero_grad(set_to_none=True)  # 清空梯度（set_to_none=True更高效）
    u_pred = model(coords)  # 模型预测：[1, N, N]

    # 1. 分域数据损失（中心区域权重更高，因为结构更复杂）
    # 中心区域损失
    data_loss_center = torch.mean(torch.abs(u_pred[0, center_mask] - targets[0, center_mask]) ** 2)
    # 边缘区域损失
    data_loss_edge = torch.mean(torch.abs(u_pred[0, edge_mask] - targets[0, edge_mask]) ** 2)
    loss_data = 1.5 * data_loss_center + data_loss_edge  # 合并数据损失

    # 2. 物理损失（ASPINN + gPINN约束）
    residual, res_grad_x, res_grad_y = nlse_residual(u_pred)  # 计算残差
    aspinn_w = aspinn_weight(residual)  # 自适应采样权重
    loss_phys = torch.mean(aspinn_w * torch.abs(residual) ** 2)  # ASPINN物理损失
    # gPINN残差梯度损失（确保残差空间平滑）
    loss_gpinn = torch.mean(torch.abs(res_grad_x) ** 2 + torch.abs(res_grad_y) ** 2)

    # 3. 振幅尺度损失（确保重建振幅的整体能量与真实值匹配）
    loss_scale = torch.abs(torch.mean(torch.abs(u_pred)) - true_amp_mean)

    # 4. 相位平滑损失（增强相位的空间连续性，减少噪声）
    pred_phase = torch.angle(u_pred)  # 提取预测波场的相位
    # 计算x和y方向的相位梯度（用差分近似，prepend保持尺寸一致）
    phase_grad_x = torch.abs(torch.diff(pred_phase, dim=2, prepend=pred_phase[..., :1]))
    phase_grad_y = torch.abs(torch.diff(pred_phase, dim=1, prepend=pred_phase[:, :1, :]))
    loss_phase_smooth = torch.mean(phase_grad_x + phase_grad_y)  # 平均梯度作为损失

    # 5. 总损失计算（带热身机制的权重调度）
    # 物理约束权重：热身阶段（<warmup_epochs）为0，之后逐渐增加到1
    alpha = min(1.0, (epoch - warmup_epochs) / weight_ramp_epochs) if epoch >= warmup_epochs else 0.0
    w_data, w_phys = adaptive_weights(loss_data, loss_phys, loss_scale)  # 自适应权重
    total_loss = loss_data + alpha * (
            w_phys * (loss_phys + 0.1 * loss_gpinn) +  # 物理约束项（含gPINN）
            0.2 * loss_scale +  # 尺度约束项
            lambda_phase_smooth * loss_phase_smooth  # 相位平滑约束项
    )

    # 反向传播与优化
    if not torch.isfinite(total_loss):  # 检查损失是否为有限值（避免NaN）
        print(f"❌ NaN at epoch {epoch}")
        break
    total_loss.backward(retain_graph=True)  # 计算梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪（防止梯度爆炸）
    optimizer.step()  # 更新参数
    scheduler.step()  # 更新学习率

    # 记录损失
    loss_log['total'].append(total_loss.item())
    loss_log['data'].append(loss_data.item())
    loss_log['phys'].append(loss_phys.item())
    loss_log['gpinna'].append(loss_gpinn.item())
    loss_log['scale'].append(loss_scale.item())
    loss_log['phase_smooth'].append(loss_phase_smooth.item())

    # 更新进度条描述（显示关键指标）
    pbar.set_description(
        f"[{epoch}] Total: {total_loss.item():.2e} | Data: {loss_data.item():.2e} | "
        f"Phys: {loss_phys.item():.2e} | PhaseSmooth: {loss_phase_smooth.item():.2e} | "
        f"AmpMean: {torch.mean(torch.abs(u_pred)).item():.3f}"
    )

# ---------- 评估与期刊级可视化 ----------
model.eval()  # 切换到评估模式
with torch.no_grad():  # 禁用梯度计算
    u_final = model(coords)[0].cpu().numpy()  # 最终预测结果（转为NumPy数组）
    U_true_np = U_true.detach().cpu().numpy()  # 真实波场
    U_noisy_np = U_noisy.detach().cpu().numpy()  # 带噪声的输入
    residual, _, _ = nlse_residual(model(coords))
    residual_np = np.abs(residual.detach().cpu().numpy())  # 物理残差

    # 将坐标转换为numpy数组（用于可视化）
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    x_min, x_max = x_np.min(), x_np.max()
    y_min, y_max = y_np.min(), y_np.max()

    # 计算定量评估指标（均方误差MSE）
    mse_amp = np.mean((np.abs(u_final) - np.abs(U_true_np)) ** 2)  # 振幅MSE
    mse_pha = np.mean((np.angle(u_final) - np.angle(U_true_np)) ** 2)  # 相位MSE
    amp_mean_error = np.abs(np.mean(np.abs(u_final)) - true_amp_mean)  # 振幅均值误差
    res_mean = np.mean(residual_np)  # 物理残差均值
    res_std = np.std(residual_np)  # 物理残差标准差

    # 打印定量结果（期刊论文级精度）
    print("\n" + "=" * 50)
    print("定量重建结果（Quantitative Reconstruction Results）")
    print("=" * 50)
    print(f"振幅均方误差（Amplitude MSE）: {mse_amp:.6f}")
    print(f"相位均方误差（Phase MSE）:     {mse_pha:.6f}")
    print(f"振幅均值误差（Amp Mean Error）: {amp_mean_error:.6f}")
    print(f"物理残差均值（Residual Mean）:  {res_mean:.4e}")
    print(f"物理残差标准差（Residual Std）: {res_std:.4e}")
    print("=" * 50 + "\n")

# 1. 分域掩码与误差分析图（期刊级布局）
fig = plt.figure(figsize=(10, 3))  # 尺寸在创建图形时设置
gs = gridspec.GridSpec(1, 3, wspace=0.3)

# 子图1：分域掩码
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(center_mask.cpu().numpy(), cmap='binary', origin='lower',
                 extent=[x_min, x_max, y_min, y_max])
ax1.set_xlabel("x (μm)")
ax1.set_ylabel("y (μm)")
ax1.set_title("Domain Mask")
cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1])
cbar1.ax.set_yticklabels(['Edge', 'Center'])
ax1.text(-0.1, 1.1, 'a', transform=ax1.transAxes, fontweight='bold')

# 子图2：物理残差（高值表示不符合物理规律的区域）
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(residual_np, cmap='plasma', origin='lower',
                 extent=[x_min, x_max, y_min, y_max])
ax2.set_xlabel("x (μm)")
ax2.set_ylabel("y (μm)")
ax2.set_title("NLSE Residual")
plt.colorbar(im2, ax=ax2)
ax2.text(-0.1, 1.1, 'b', transform=ax2.transAxes, fontweight='bold')

# 子图3：振幅误差
ax3 = fig.add_subplot(gs[0, 2])
amp_error = np.abs(u_final) - np.abs(U_true_np)
im3 = ax3.imshow(amp_error, cmap='bwr', origin='lower',
                 extent=[x_min, x_max, y_min, y_max],
                 vmin=-0.1, vmax=0.1)  # 限制范围，突出细节
ax3.set_xlabel("x (μm)")
ax3.set_ylabel("y (μm)")
ax3.set_title("Amplitude Error")
plt.colorbar(im3, ax=ax3)
ax3.text(-0.1, 1.1, 'c', transform=ax3.transAxes, fontweight='bold')

plt.tight_layout()
# 保存为1000 dpi的PNG，移除无效的figsize参数
plt.savefig(f"{save_dir}/domain_residual_error.png", dpi=1000)
plt.show()

# 2. 重建结果对比图（带局部放大）
fig = plt.figure(figsize=(10, 6))  # 尺寸在创建图形时设置
gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.3)

# 振幅子图（第一行）
cmap_amp = plt.cm.viridis  # 期刊推荐的颜色映射（避免jet的视觉偏差）
titles_amp = ["True Amplitude", "Noisy Input", "Reconstructed Amplitude"]
amps = [np.abs(U_true_np), np.abs(U_noisy_np), np.abs(u_final)]

# 计算放大区域的索引范围（使用切片而非布尔索引，确保维度匹配）
zoom_x_min, zoom_x_max = -1, 1
zoom_x_start = np.argmin(np.abs(x_np - zoom_x_min))
zoom_x_end = np.argmin(np.abs(x_np - zoom_x_max)) + 1  # +1确保包含终点
zoom_y_min, zoom_y_max = -0.5, 0.5
zoom_y_start = np.argmin(np.abs(y_np - zoom_y_min))
zoom_y_end = np.argmin(np.abs(y_np - zoom_y_max)) + 1  # +1确保包含终点

for i in range(3):
    ax = fig.add_subplot(gs[0, i])
    im = ax.imshow(amps[i], cmap=cmap_amp, origin='lower',
                   extent=[x_min, x_max, y_min, y_max])
    ax.set_xlabel("x (μm)")
    if i == 0:
        ax.set_ylabel("y (μm)")
    ax.set_title(titles_amp[i])
    plt.colorbar(im, ax=ax, shrink=0.8)
    if i == 2:
        # 绘制放大区域边框
        rect = plt.Rectangle(
            (zoom_x_min, zoom_y_min), zoom_x_max - zoom_x_min, zoom_y_max - zoom_y_min,
            edgecolor='white', linestyle='--', fill=False, linewidth=1.5
        )
        ax.add_patch(rect)
        # 创建局部放大图
        axins = inset_axes(ax, width="30%", height="30%", loc='upper left')
        axins.imshow(
            amps[2][zoom_y_start:zoom_y_end, zoom_x_start:zoom_x_end],
            cmap=cmap_amp,
            origin='lower'
        )
        axins.set_title(
            "Zoomed",
            color='red',
            fontsize=7,
            pad=1,
            y=0.95
        )
        axins.set_xticks([])
        axins.set_yticks([])
    ax.text(-0.1, 1.1, chr(97 + i), transform=ax.transAxes, fontweight='bold')

# 相位子图（第二行）
cmap_pha = plt.cm.twilight
titles_pha = ["True Phase", "Noisy Phase", "Reconstructed Phase"]
phas = [np.angle(U_true_np), np.angle(U_noisy_np), np.angle(u_final)]

for i in range(3):
    ax = fig.add_subplot(gs[1, i])
    im = ax.imshow(phas[i], cmap=cmap_pha, origin='lower',
                   extent=[x_min, x_max, y_min, y_max],
                   vmin=-np.pi, vmax=np.pi)
    ax.set_xlabel("x (μm)")
    if i == 0:
        ax.set_ylabel("y (μm)")
    ax.set_title(titles_pha[i])
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[-np.pi, 0, np.pi])
    cbar.ax.set_yticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    ax.text(-0.1, 1.1, chr(97 + 3 + i), transform=ax.transAxes, fontweight='bold')

plt.tight_layout()
# 保存为1000 dpi的PNG
plt.savefig(f"{save_dir}/reconstruction_results.png", dpi=1000)
plt.show()

# 3. 损失曲线可视化（半对数坐标，突出收敛趋势）
fig = plt.figure(figsize=(8, 6))  # 尺寸在创建图形时设置
gs = gridspec.GridSpec(2, 1, hspace=0.3)

# 总损失与分项损失
ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogy(loss_log['total'], label='Total Loss', color='k')
ax1.semilogy(loss_log['data'], label='Data Loss', color='C0', linestyle='--')
ax1.semilogy(loss_log['phys'], label='Physics Loss', color='C1', linestyle='-.')
ax1.axvline(x=warmup_epochs, color='gray', linestyle=':', label='Physics Onset')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Convergence')
ax1.legend(loc='upper right')
ax1.text(-0.1, 1.1, 'a', transform=ax1.transAxes, fontweight='bold')

# 相位平滑与尺度损失
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(loss_log['phase_smooth'], label='Phase Smoothness', color='C2')
ax2.plot(loss_log['scale'], label='Amplitude Scale', color='C3')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(loc='upper right')
ax2.text(-0.1, 1.1, 'b', transform=ax2.transAxes, fontweight='bold')

plt.tight_layout()
# 保存为1000 dpi的PNG
plt.savefig(f"{save_dir}/loss_curves.png", dpi=1000)
plt.show()

# 4. 物理残差与波场结构关联性分析
fig = plt.figure(figsize=(8, 4))  # 尺寸在创建图形时设置
gs = gridspec.GridSpec(1, 2, wspace=0.3)

# 残差空间分布（叠加振幅轮廓）
ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(residual_np, cmap='plasma', origin='lower',
                extent=[x_min, x_max, y_min, y_max])
# 叠加真实振幅轮廓线
X_np, Y_np = np.meshgrid(x_np, y_np, indexing='ij')
cont = ax1.contour(X_np, Y_np, np.abs(U_true_np), levels=5,
                   colors='white', linestyles='--', linewidths=1)
ax1.clabel(cont, inline=True, fontsize=6)
ax1.set_xlabel("x (μm)")
ax1.set_ylabel("y (μm)")
ax1.set_title("Residual with Amplitude Contours")
plt.colorbar(im, ax=ax1)
ax1.text(-0.1, 1.1, 'a', transform=ax1.transAxes, fontweight='bold')

# 残差与振幅梯度的相关性分析
ax2 = fig.add_subplot(gs[0, 1])
amp_grad = np.gradient(np.abs(U_true_np))
amp_grad_mag = np.sqrt(amp_grad[0] ** 2 + amp_grad[1] ** 2)
corr_coef = np.corrcoef(residual_np.flatten(), amp_grad_mag.flatten())[0, 1]
ax2.scatter(amp_grad_mag.flatten(), residual_np.flatten(),
            s=1, alpha=0.3, color='C4')
ax2.set_xlabel("Amplitude Gradient (a.u.)")
ax2.set_ylabel("Residual Magnitude")
ax2.set_title(f"Residual vs. Gradient (r = {corr_coef:.2f})")
ax2.text(-0.1, 1.1, 'b', transform=ax2.transAxes, fontweight='bold')

plt.tight_layout()
# 保存为1000 dpi的PNG
plt.savefig(f"{save_dir}/residual_correlation.png", dpi=1000)
plt.show()

# 保存定量结果到文本文件
with open(f"{save_dir}/quantitative_results.txt", "w") as f:
    f.write("Quantitative Reconstruction Metrics\n")
    f.write("===================================\n")
    f.write(f"Amplitude MSE:        {mse_amp:.6f}\n")
    f.write(f"Phase MSE:            {mse_pha:.6f}\n")
    f.write(f"Amplitude Mean Error: {amp_mean_error:.6f}\n")
    f.write(f"Residual Mean:        {res_mean:.4e}\n")
    f.write(f"Residual Std:         {res_std:.4e}\n")
    f.write(f"Correlation Coefficient: {corr_coef:.4f}\n")

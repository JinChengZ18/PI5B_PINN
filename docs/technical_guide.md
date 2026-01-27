# PINN 技术文档

本文档包含PINN项目的完整技术细节、原理推导、架构设计和扩展指南。

[TOC]



---

## 1. PINN理论基础

### 1.1 物理信息神经网络概述

物理信息神经网络(Physics-Informed Neural Networks, PINN)是一种将物理定律嵌入到深度学习框架中的方法，由Raissi等人于2019年提出。

**核心思想**:
```
传统DL:  数据 → 神经网络 → 预测
PINN:    数据 + 物理方程 → 神经网络 → 符合物理规律的预测
```

### 1.2 稳态热传导方程

本项目求解稳态热传导问题，控制方程为:

```
∇·(k∇T) + Q = 0
```

在热导率 `k` 为常数时简化为:

```
k∇²T + Q = 0

其中:
∇²T = ∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²  (拉普拉斯算子)
```

**物理意义**:
- `∇²T`: 温度场的二阶空间导数，表示热扩散
- `Q`: 体积热源 (W/m³)，如芯片发热
- 方程含义: 热传导与热源平衡

### 1.3 PINN求解方法

#### 传统方法 vs PINN

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| FEM (COMSOL) | 网格离散化 + 数值求解 | 精度高，通用性强 | 计算慢，参数扫描耗时 |
| PINN | 神经网络拟合 + 物理约束 | 快速推理，参数化能力强 | 需要训练数据 |

#### PINN损失函数

```python
L_total = L_data + λ_physics × L_physics

其中:
L_data = MSE(T_pred, T_COMSOL)          # 数据拟合损失
L_physics = mean(|∇²T_pred|²)           # 物理约束损失
```

**关键**:
- `L_data`: 确保预测与COMSOL仿真数据一致
- `L_physics`: 确保预测满足热传导方程
- `λ_physics`: 平衡两者的权重
            
---

## 2. 网络架构详解

### 2.1 ThermalPINN整体结构

```

```

### 2.2 Fourier特征编码

#### 原理

神经网络对低频信号拟合较好，但对高频信号(如温度梯度)拟合困难。Fourier特征编码通过将输入映射到频域，显著提升高频拟合能力。

#### 数学公式

```python
γ(v) = [sin(2πB₁v), cos(2πB₁v), ..., sin(2πBₙv), cos(2πBₙv)]

其中:
v: 输入向量 (x, y, z, soc_power, ...)
B: 频率矩阵 (n × 7), 从高斯分布 N(0, σ²) 采样
σ: fourier_sigma (本项目取1.0)
```

#### 实现

```python

```

**参数说明**:
- `in_dim = 7`: 输入维度 (3坐标 + 4热源参数)
- `fourier_dim = 192`: 频率数量
- 输出维度: `192 × 2 = 384`

**频率分布示例**:
```
σ = 1.0 → 频率集中在 [0.1, 10] 范围 (适合mm级坐标)
σ = 0.5 → 低频为主 (平滑温度场)
σ = 2.0 → 高频为主 (锐利梯度)
```





---

## 3. 物理损失函数

### 3.1 拉普拉斯算子计算

#### 数学定义

```
∇²T = ∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²
```

#### 自动微分实现

```python

```

**计算复杂度**:
- 前向传播: 1次
- 一阶导数: 1次反向传播
- 二阶导数: 3次反向传播 (x, y, z各一次)
- **总计**: ~5倍单次前向传播的时间

### 3.2 物理损失函数

```python
# model/physics_loss.py 第82-125行
class HeatEquationLoss(nn.Module):
    def __init__(self, k=1.0):
        super().__init__()
        self.k = k  # 热导率 (假设为常数)

    def forward(self, laplacian_T):
        # 热传导方程: k∇²T = 0 (忽略体积热源Q)
        # 物理损失: L = mean(|∇²T|²)
        physics_loss = torch.mean(laplacian_T ** 2)
        return physics_loss
```

**注意**:
- 本项目未显式建模体积热源 `Q`
- 而是通过边界条件和数据拟合隐式学习
- 因此物理损失简化为 `||∇²T||²`

### 3.3 组合损失函数

```python

```

**权重选择**:
- `λ_data = 1.0`: 数据拟合为主要目标
- `λ_physics = 0.02`: 物理约束为辅助 (防止过拟合，提升泛化)

**权重调优建议**:
```
λ_physics 过小 (0.001) → 物理约束弱，可能违反物理规律
λ_physics 适中 (0.01-0.05) → 平衡，推荐
λ_physics 过大 (0.5) → 物理约束过强，拟合困难
```

---

## 4. 训练策略



---

## 5. 配置系统详解

### 5.1 配置文件结构

[options/train_pinn.yml](../options/train_pinn.yml) 完整结构:

```yaml

```

### 5.2 参数影响分析

#### 网络容量参数

| 参数 | 影响 | 推荐范围 | 默认值 |
|------|------|---------|--------|
| `hidden_dim` | 模型容量 ↑ 精度 ↑ 训练时间 ↑ | 256-512 | 384 |
| `depth` | 网络深度 ↑ 表达能力 ↑ 但需残差连接 | 6-12 | 9 |
| `fourier_dim` | 频率特征 ↑ 高频拟合 ↑ | 128-256 | 192 |
| `dropout` | 正则化 ↑ 防止过拟合 | 0.05-0.2 | 0.1 |

#### 训练参数

| 参数 | 影响 | 推荐范围 | 默认值 |
|------|------|---------|--------|
| `batch_size` | 越大越稳定，但显存占用 ↑ | 2-8 | 2 |
| `lr` | 学习率 ↑ 收敛快但可能不稳定 | 1e-4 ~ 1e-3 | 5e-4 |
| `lambda_physics` | 物理约束 ↑ 泛化 ↑ 但拟合难度 ↑ | 0.01-0.1 | 0.02 |







---

## 🔗 外部资源

### PINN理论

- [Raissi et al., 2019 - Physics-Informed Neural Networks](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [PINN综述 (Cuomo et al., 2022)](https://link.springer.com/article/10.1007/s10915-022-01939-z)

### Fourier特征

- [Tancik et al., NeurIPS 2020 - Fourier Features](https://arxiv.org/abs/2006.10739)

### PyTorch文档

- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [自动微分 (Autograd)](https://pytorch.org/docs/stable/autograd.html)

---







---

**文档更新**: 2025-12-30

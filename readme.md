# PINN 树莓派热场预测系统

> 基于物理信息神经网络的参数化热场快速预测工具

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 项目简介

本项目实现了基于**物理信息神经网络(Physics-Informed Neural Networks, PINN)** 的树莓派5热场预测系统。通过训练深度神经网络，能够根据不同的热源功率组合，**即时预测**整个PCB板的温度分布，**无需运行耗时的COMSOL仿真**。

### 核心特性

- 🔬 **物理约束**: 集成热传导方程(∇²T=0)作为物理损失，确保预测符合物理规律
- ⚡ **快速推理**: 预测时间 < 1秒，相比COMSOL仿真加速 1000+ 倍
- 🎯 **高精度**: 与COMSOL仿真对比，平均绝对误差 < 0.5 K
- 🔧 **参数化**: 支持4个热源功率组合(SoC, PMIC, USB, Other)，可扩展至风扇、几何等参数
- 📊 **模块化设计**: 易于扩展到多物理场耦合

### 应用场景

1. **热设计优化**: 快速评估不同功耗配置下的温度分布
2. **参数扫描**: 替代耗时的COMSOL参数扫描
3. **实时预测**: 嵌入式系统运行时温度预测
4. **逆向设计**: 给定温度约束，反推最优功耗配置



---

## 快速开始

### 1. 安装依赖

```bash
pip install requirements.txt
```

### 2. 准备数据

由于数据规模较大（~6GB），未直接包含在仓库中。

请按以下步骤获取数据：

1. 安装云存储工具（如 ossutil）
2. 运行数据下载脚本：

```bash
python scripts/download_data.py
```

数据将被下载至`data/thermal_heat_source/`，确保COMSOL仿真数据已正确下载后:

```bash
# 生成数据索引
python scripts/generate_index.py

# 验证数据
# 应看到: "找到 136 个 CSV 文件"
```

详见: [数据指南](docs/data_guide.md)

### 3. 训练模型

```bash
python model/train.py
```

**训练时间**:
- GPU (RTX 5060): ~0.5-2小时
- CPU: ~4-5小时

**训练输出**:
```

```

**日志文件**: 自动保存到 `checkpoints/` 目录。

### 4. 选用不同的模型版本

换用不同版本的model模型，可在`versions`目录下寻找，将其复制到主目录下即可使用。

详见: [版本指南](docs/version.md)



---

## 项目结构

```
comsol_pinn/
├── model/                      # PINN核心模块
│   ├── model.py                # ThermalPINN网络
│   ├── dataset.py              # 数据集加载器
│   └── train.py                # 训练脚本
│
├── data/                       # 数据目录
│   └── thermal_heat_source/    # 热源参数化数据集 (144个案例)
│       ├── index.jsonl         # 数据索引
│       └── *.csv               # COMSOL导出数据
│
├── scripts/                    # 工具脚本
│   ├── generate_index.py       # 生成数据索引
│   ├── check_data_quality.py   # 数据质量检查
│   ├── check_data_stat.py      # 数据统计分布
│   ├── vis_data.py             # 可视化 (分数据集)
│   └── vis_data_stat.py        # 可视化 (数据集层次聚合)
│
├── checkpoints/                # 模型检查点 (训练后生成)
│   └── thermal_pinn/
│       ├── best_model.pt       # 最佳模型
│       ├── train_*.log         # 训练日志（带时间戳）
│       └── config_*.yml        # 配置快照
│
├── docs/                       # 文档
│   ├── technical_guide.md      # 技术详解 (PINN原理、架构设计)
│   ├── data_guide.md           # 数据指南 (格式、处理、质量检查)
│   └── logging_guide.md        # 日志系统使用指南
│
└── requirements.txt            # Python依赖
```



---

## 技术架构

本项目基于物理信息神经网络(PINN)，集成Fourier特征编码、残差连接和物理约束损失函数，实现高精度温度场预测。详细的PINN原理、网络架构、数学推导请参考: [技术指南](docs/technical_guide.md)

---

### 性能指标

#### 模型性能 (RTX 5060 8GB)

| 指标 | 值 |
|------|---|
| 参数量 | ~2.5M |
| 训练时间 | 2.5-3.5 小时 |
| 推理速度 | < 1 秒/案例 |
| 显存占用 | ~5-6 GB |

#### 预测精度 (验证集)

| 指标 | 目标 | 实测 |
|------|------|------|
| MAE | < 0.5 K | ~0.35 K |
| RMSE | < 1.0 K | ~0.57 K |
| MAPE | < 1% | ~0.12% |
| Max Error | < 3.0 K | ~2.1 K |




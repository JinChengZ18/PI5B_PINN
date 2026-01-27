# 数据指南

本文档详细说明PINN训练数据的格式、生成流程、质量检查和故障排除方法。

---

[TOC]

---

## 1. 数据目录结构

```
data/
├── thermal_heat_source/          # 热源参数化数据集 (当前)
│   ├── index.jsonl               # 数据索引文件 ⭐(储存功率参数)
│   └── case_****.csv             # COMSOL导出的温度场数据 (*表示对应的数字序号)
│
├── thermal_fan/                  # 风扇参数化数据集 (未来扩展)
│   ├── index.jsonl
│   └── *.csv
│
└── thermal_geometry/             # 几何参数化数据集 (未来扩展)
    ├── index.jsonl
    └── *.csv
```

#### 当前数据集: thermal_heat_source

| 属性 | 值 |
|------|---|
| 数据集类型 | 热源参数化 |
| 案例数量 | 144 个 |
| 单文件大小 | ~40 MB |
| 总数据大小 | ~5.7 GB |
| 每个案例点数 | ~52万个空间点 |
| 参数化维度 | 4 (热源功率) |



---

## 2. 数据格式详解

### 2.1 CSV文件格式

#### 文件结构

COMSOL导出的CSV文件包含两部分：
1. **元数据注释** (以 `%` 开头)
2. **数据内容** (坐标和温度)

#### 示例文件

```csv
% Model,case0001.mph
% Version,COMSOL 6.3.0.290
% Date,"Dec 22 2025, 11:37"
% Dimension,3
% Nodes,523075
% Expressions,2
% Description,"温度, 速度大小"
% Length unit,mm
% x,y,z,T (K),spf.U (m/s)
48.49999999999999,64.5,30.999999999999996,296.8066996195429,NaN
46.76303980543846,64.21983877323764,32.5,296.80676055990574,NaN
50.00000000000001,64.41407092983208,32.5,296.80724878707764,NaN
...
```

#### 列说明

| 列名 | 单位 | 含义 |
|------|------|------|
| `x` | mm | X坐标 |
| `y` | mm | Y坐标 |
| `z` | mm | Z坐标 |
| `T (K)` | K | 温度 |
| `spf.U (m/s)` | m/s | 速度 (未使用) |

### 2.2 index.jsonl格式

#### 文件作用

`index.jsonl` 是数据索引文件，记录每个CSV文件的参数和元数据。

#### 格式说明

JSONL (JSON Lines) 格式，每行一个JSON对象：

```jsonl

```

#### 字段说明

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `case_id` | string | 案例ID | "case_0001" |
| `export_file` | string | CSV文件相对路径 | ".../case_0001.csv" |
| `parameters` | object | 参数化配置 | `{"soc_power": 2.0, ...}` |
| `metadata` | object | 元数据 | `{"model_file": "...", ...}` |

**parameters字段详解**:

| 参数名 | 单位 | 含义 | 可选值 |
|--------|------|------|--------|
| `soc_power` | W | SoC芯片功耗 | [2.0, 6.0, 10.0, 12.0] |
| `pmic_power` | W | 电源管理IC功耗 | [0.2, 0.5, 1.0, 1.5] |
| `usb_power` | W | USB接口功耗 | [0.1, 0.3, 0.5] |
| `other_power` | W | 其他器件功耗 | [0.2, 0.5, 1.0] |

**参数空间**:
- 总组合数: 4 × 4 × 3 × 3 = 144 个
- 实际数据: 144 个 (全覆盖)



---

## 3. 数据生成流程

### 3.1 COMSOL仿真配置

#### 热源参数配置文件

[options/heat_sources.yml](../options/heat_sources.yml):

```yaml
metadata_templates:
  tag: "SoC_{soc_power:.1f}W_PMIC_{pmic_power:.1f}W_USB_{usb_power:.1f}W_Other_{other_power:.1f}W"
metadata_expressions:
  total_power: "soc_power + pmic_power + usb_power + other_power"

# 四个热源功率参数化配置
# 约束条件: SoC功率 > PMIC功率 > USB功率, Other功率
parameters:
  - name: soc_power
    unit: W
    alias: soc_power
    description: BCM2712 SoC heat dissipation power (主芯片，最大热源)
    values: [2.0, 6.0, 10.0, 12.0]  # 空闲、轻负载、高负载、峰值TDP

  - name: pmic_power
    unit: W
    alias: pmic_power
    description: PMIC (Power Management IC) heat dissipation power (电源管理芯片)
    values: [0.2, 0.5, 1.0, 1.5]  # 次要热源

  - name: usb_power
    unit: W
    alias: usb_power
    description: USB and Ethernet interface heat dissipation power (USB/网络接口)
    values: [0.1, 0.3, 0.5]  # 较小热源

  - name: other_power
    unit: W
    alias: other_power
    description: Other component heat dissipation power (其他器件，如RP1南桥)
    values: [0.2, 0.5, 1.0]  # 较小热源
```

### 3.2 生成数据步骤

#### 步骤1: 运行COMSOL批处理

```bash
# 使用MATLAB接口运行COMSOL参数扫描
python -m auto_param_data/cli.py --config options/heat_sources.yml
```

**自动化流程**:
1. 读取参数配置文件
2. 生成所有参数组合 (144个)
3. 逐个运行COMSOL仿真
4. 导出温度场为CSV

**预计时间**: ~2-3小时/案例 × 144 = 6-9天 (并行可缩短)

#### 步骤2: 生成索引文件

```bash
python scripts/generate_index.py
```

**功能**:
- 扫描 `data/thermal_heat_source/` 目录
- 解析CSV文件名
- 提取参数映射
- 生成 `index.jsonl`



---

## 4. 数据质量检查

```
├── scripts/          # 热源参数化数据集 (当前)
│   ├── check_data_quality.py                 # 数据质量检查脚本 ⭐
│   ├── check_data_stat.py                    # 数据统计脚本
│   ├── vis_data.py                           # 数据可视化脚本
│   └── vis_data_stat.py                      # 数据可视化脚本 (层次聚合)
```

### 4.1 自动质量检查

运行质量检查脚本:

```bash
python scripts/check_data_quality.py
```

**检查项**:

- [ ] CSV文件完整性 (数量、大小)
- [ ] 列名一致性
- [ ] 数值范围有效性
- [ ] NaN/Inf检查
- [ ] 坐标网格均匀性
- [ ] index.jsonl完整性

### 4.2 数据统计信息

运行数据统计脚本:

```bash
python scripts/check_data_stat.py
```

期望输出：

```
============================================================
PINN 数据统计信息
============================================================
发现 141 个 CSV 文件，开始统计...

=== 每列统计信息 ===
列名                       min          max         mean          std      NaN      Inf
--------------------------------------------------------------------------------
x                   -50.0000      50.0000      -9.2951      23.0280        0        0
y                    -6.5000      68.5000      26.4142      19.4133        0        0
z                   -32.5000      32.5000       1.0515      15.2256        0        0
T (K)               289.8997     494.9643     321.0174      27.1489        0        0
spf.U (m/s)           0.0000       1.4612       0.0494       0.1208 66092904        0

统计完成！
============================================================
```

### 4.3 可视化检查

运行数据可视化脚本:

```bash
python scripts/vis_data.py
```

期望输出：

<img src="C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20251226121945628.png" alt="image-20251226121945628" style="zoom:50%;" />

以上输出为每个对应温度场数据集 `(x,y,z,T)` 的三维散点可视化。

### 4.4 数据合理性检查

运行高级数据可视化脚本:

```bash
python scripts/vis_data_stat.py
```

该脚本按照 `index.jsonl` 中的 `soc_power`、`pmic_power`、`usb_power`、`other_power` 参数，对原始数据集进行分层聚合，以形成对应参数下的数据集平均结果，输出为对应的csv文件，并根据这些文件形成温度场随某一参数变化情况的gif文件。

<img src="C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20251230160731899.png" alt="image-20251230160731899" style="zoom: 50%;" />



<div style="page-break-after: always;"></div>

---

## 附录

### A. 数据集统计

#### 当前数据集 (thermal_heat_source)

```
总案例数: 144
参数化维度: 4
  - soc_power: [2.0, 6.0, 10.0, 12.0] W (4个值)
  - pmic_power: [0.2, 0.5, 1.0, 1.5] W (4个值)
  - usb_power: [0.1, 0.3, 0.5] W (3个值)
  - other_power: [0.2, 0.5, 1.0] W (3个值)

总功率范围: 2.5 W ~ 15.0 W
温度范围: 296.0 K ~ 330.0 K (23°C ~ 57°C)

空间采样:
  - X方向: [0, 85] mm
  - Y方向: [0, 56] mm
  - Z方向: [0, 50] mm
  - 点数/案例: ~52万个

文件统计:
  - CSV文件数: 136
  - 单文件大小: ~40 MB
  - 总数据大小: ~5.7 GB
```

### B. 参数敏感度分析

根据COMSOL仿真结果，各参数对最高温度的影响:

| 参数 | 影响权重 | 说明 |
|------|---------|------|
| `soc_power` | ⭐⭐⭐⭐⭐ | SoC是主要热源，影响最大 |
| `pmic_power` | ⭐⭐⭐ | PMIC靠近SoC，有显著影响 |
| `usb_power` | ⭐⭐ | USB距离SoC较远，影响较小 |
| `other_power` | ⭐ | 其他器件分散，影响最小 |

**结论**: PINN模型需要重点关注 `soc_power` 和 `pmic_power` 的拟合精度。

### C. 未来数据集扩展

#### thermal_fan (风扇参数化)

```
新增参数:
  - fan_speed: [0, 3000, 5000] RPM (3个值)
  - fan_pos_x: [20, 42.5, 65] mm (3个值)
  - fan_pos_y: [28] mm (1个值)
  - fan_pos_z: [45] mm (1个值)

预计案例数: 144 × 3 × 3 = 1296 个
预计数据大小: ~50 GB

新增输出:
  - 速度场 (vx, vy, vz)
```

#### thermal_geometry (几何参数化)

```
新增参数:
  - heatsink_height: [10, 25, 40] mm (3个值)
  - fin_spacing: [1, 2, 3] mm (3个值)
  - fin_thickness: [0.5, 1.0, 1.5] mm (3个值)

预计案例数: 144 × 3 × 3 × 3 = 3888 个
预计数据大小: ~150 GB

技术挑战:
  - 动态网格
  - 几何编码
```

---

**文档更新**: 2025-12-08

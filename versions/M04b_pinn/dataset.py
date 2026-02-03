# dataset.py
import json
import csv
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys


# ==========================================================
# 项目根目录（返回上级目录）
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 数据目录
DATA_DIR = PROJECT_ROOT / "data" / "thermal_heat_source"
INDEX_FILE = DATA_DIR / "index.jsonl"


def read_comsol_csv(csv_path):
    """
    读取 COMSOL CSV（最终稳定版）
    规则：
    1. 从 % 注释行中解析 header
    2. 如果未找到 header，则使用默认列顺序
    3. 忽略 spf.U 等无关列
    """
    data = []
    header = None

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue

            # ---------- 1. 从注释中解析 header ----------
            if row[0].startswith("%"):
                line = row[0].lstrip("%").strip().lower()
                if "x" in line and "y" in line and "z" in line and "t" in line:
                    header = [h.strip() for h in line.split(",")]
                continue

            # ---------- 2. 如果 header 还没解析，说明 COMSOL 没给 ----------
            if header is None:
                # COMSOL 默认顺序：x, y, z, T, spf.U
                header = ["x", "y", "z", "t", "unused"]

            # ---------- 3. 建立列索引 ----------
            def find_col(keywords):
                for i, h in enumerate(header):
                    for kw in keywords:
                        if kw in h:
                            return i
                raise ValueError(f"无法在 CSV 中找到列 {keywords}, header={header}")

            col_x = find_col(["x"])
            col_y = find_col(["y"])
            col_z = find_col(["z"])
            col_t = find_col(["t"])

            # ---------- 4. 读取数据 ----------
            try:
                x = float(row[col_x])
                y = float(row[col_y])
                z = float(row[col_z])
                T = float(row[col_t])
                data.append([x, y, z, T])
            except ValueError:
                continue  # 跳过 NaN / 非法行

    if len(data) == 0:
        raise RuntimeError(f"CSV 文件未读取到任何有效数据: {csv_path}")

    return torch.tensor(data, dtype=torch.float32)



class ThermalHeatSourceDataset(Dataset):
    """
    数据集：
    - 每个 item = 一个 case 的分层采样点
    """
    def __init__(self,
        points_per_case=4096,
        hot_ratio=0.4,        # 热区抽样比例（0~1）
        temp_power=3.0,       # 温度权重幂次，越大越偏向热区
        eps=1e-8,
    ):
        self.points_per_case = points_per_case
        self.hot_ratio = hot_ratio
        self.temp_power = temp_power
        self.eps = eps

        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            self.index = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        case = self.index[idx]
        csv_path = Path(case["export_file"])
        data = read_comsol_csv(csv_path)  # (N,4)

        N = data.shape[0]
        B = min(self.points_per_case, N)
        # --- 1) uniform 采样 ---
        B_hot = int(B * self.hot_ratio)
        B_uni = B - B_hot
        ids_uni = torch.randperm(N)[:B_uni]
        # --- 2) hot-biased 采样（按温度加权）---
        T_all = data[:, 3]  # (N,)
        T_min = T_all.min()
        T_max = T_all.max()
        T_norm = (T_all - T_min) / (T_max - T_min + self.eps)  # [0,1]
        # 权重：让高温点概率更大
        w = (T_norm + self.eps) ** self.temp_power
        w = w / w.sum()
        # multinomial 默认可重复抽样；这里设 replacement=False 更合理
        ids_hot = torch.multinomial(w, num_samples=min(B_hot, N), replacement=False)
        ids = torch.cat([ids_uni, ids_hot], dim=0)
        sampled = data[ids]

        # ===== 坐标归一化到 [-1,1]（体系一致）=====
        xyz = sampled[:, :3]
        xyz_min = xyz.min(dim=0, keepdim=True)[0]
        xyz_max = xyz.max(dim=0, keepdim=True)[0]
        xyz = 2 * (xyz - xyz_min) / (xyz_max - xyz_min) - 1

        # ===== 功率参数 =====
        p = case["parameters"]
        # 原始功率参数（单位 W）
        params_raw = torch.tensor([
            p["soc_power"],
            p["pmic_power"],
            p["usb_power"],
            p["other_power"],
        ], dtype=torch.float32)
        # 推荐归一化方式：min-max 到 [0,1]，再线性映射到 [-1,1]
        params_raw_min = params_raw.min(dim=0, keepdim=True)[0]
        params_raw_max = params_raw.max(dim=0, keepdim=True)[0]
        params_norm = 2 * (params_raw - params_raw_min) / (params_raw_max - params_raw_min) - 1
        params = params_norm.unsqueeze(0).repeat(xyz.shape[0], 1)
        x7 = torch.cat([xyz, params], dim=1)

        # ===== T 归一化 =====
        T = sampled[:, 3:4]
        T_min = T.min()
        T_max = T.max()
        T_norm = 2.0 * (T - T_min) / (T_max - T_min + 1e-8) - 1.0

        return x7, T_norm, T_min, T_max




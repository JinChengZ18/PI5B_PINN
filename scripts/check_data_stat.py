"""
check_data_stat.py
统计 thermal_heat_source 数据集中每列的统计信息
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================================
# 添加项目根目录到路径（与 test_dataset 一致）
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data/thermal_heat_source"

# ==========================================================
# COMSOL CSV 读取函数（必须与 dataset 一致）
# ==========================================================
def read_comsol_csv(path):
    return pd.read_csv(
        path,
        comment="%",
        header=None,
        names=["x", "y", "z", "T (K)", "spf.U (m/s)"]
    )

# ==========================================================
# 统计单个 CSV 文件
# ==========================================================
def compute_stats(df):
    stats = {}

    for col in df.columns:
        series = df[col]

        stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "nan_count": int(series.isna().sum()),
            "inf_count": int(np.isinf(series).sum()),
        }

    return stats

# ==========================================================
# 打印整洁的统计表
# ==========================================================
def print_stats_table(stats_dict):
    print("\n=== 每列统计信息 ===")
    print(f"{'列名':<15} {'min':>12} {'max':>12} {'mean':>12} {'std':>12} {'NaN':>8} {'Inf':>8}")
    print("-" * 80)

    for col, s in stats_dict.items():
        print(f"{col:<15} "
              f"{s['min']:>12.4f} "
              f"{s['max']:>12.4f} "
              f"{s['mean']:>12.4f} "
              f"{s['std']:>12.4f} "
              f"{s['nan_count']:>8d} "
              f"{s['inf_count']:>8d}")

# ==========================================================
# 主流程
# ==========================================================
def main():
    print("=" * 60)
    print("PINN 数据统计信息")
    print("=" * 60)

    csv_files = sorted(DATA_DIR.glob("case_*.csv"))
    if not csv_files:
        print("未找到 CSV 文件")
        return

    print(f"发现 {len(csv_files)} 个 CSV 文件，开始统计...")

    # 累积统计
    all_data = {
        "x": [],
        "y": [],
        "z": [],
        "T (K)": [],
        "spf.U (m/s)": [],
    }

    for csv_file in csv_files:
        df = read_comsol_csv(csv_file)

        for col in all_data.keys():
            all_data[col].append(df[col].values)

    # 合并所有文件的数据
    merged = {col: np.concatenate(all_data[col]) for col in all_data}

    # 计算统计信息
    stats = {}
    for col, arr in merged.items():
        arr = pd.Series(arr)
        stats[col] = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "nan_count": int(arr.isna().sum()),
            "inf_count": int(np.isinf(arr).sum()),
        }

    print_stats_table(stats)

    print("\n统计完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference 结果可视化脚本（参照 data 原始数据 3D 可视化风格）
- 3D 散点图展示 T_pred
- 3D 散点图展示 T_true - T_pred
逐 case 绘制并保存 PNG
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ==========================================================
# 项目根目录
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# 论文级绘图风格（与 data 可视化保持一致）
# ==========================================================
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 200,
    "savefig.dpi": 300,
})


# ==========================================================
# 3D 散点绘制函数
# ==========================================================
def plot_3d_scatter(
    x, y, z, values,
    title,
    cbar_label,
    save_path,
    cmap,
    norm=None,
    max_points=50000
):
    if len(x) > max_points:
        idx = np.random.choice(len(x), max_points, replace=False)
        x, y, z, values = x[idx], y[idx], z[idx], values[idx]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        x, y, z,
        c=values,
        cmap=cmap,
        norm=norm,
        s=1,
        alpha=0.85
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    cb = plt.colorbar(sc, ax=ax, shrink=0.6)
    cb.set_label(cbar_label)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


# ==========================================================
# 主流程
# ==========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visualize inference results in 3D (T_pred and T_true - T_pred)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="output/M04a/thermal_heat_source",
        help="Inference CSV directory"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="vis/inference/M04a",
        help="Visualization output directory"
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Maximum number of cases to visualize"
    )
    args = parser.parse_args()

    input_dir = PROJECT_ROOT / args.input_dir
    out_dir = PROJECT_ROOT / args.out_dir
    assert input_dir.exists(), f"Input directory not found: {input_dir}"

    csv_files = sorted(input_dir.glob("case_*.csv"))
    if args.max_cases is not None:
        csv_files = csv_files[:args.max_cases]

    print(f"[INFO] Found {len(csv_files)} cases")

    # ------------------------------------------------------
    # 统一色标范围
    # ------------------------------------------------------
    all_T_pred, all_err = [], []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        T_pred = df["T_pred"].values
        T_true = df["T_true"].values
        all_T_pred.append(T_pred)
        all_err.append(T_true - T_pred)

    all_T_pred = np.concatenate(all_T_pred)
    all_err = np.concatenate(all_err)

    T_min, T_max = all_T_pred.min(), all_T_pred.max()
    err_abs_max = np.max(np.abs(all_err))

    err_norm = TwoSlopeNorm(
        vmin=-err_abs_max,
        vcenter=0.0,
        vmax=err_abs_max
    )

    # ------------------------------------------------------
    # 逐 case 绘制
    # ------------------------------------------------------
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        x = df["x"].values
        y = df["y"].values
        z = df["z"].values
        T_pred = df["T_pred"].values
        T_true = df["T_true"].values
        err = T_true - T_pred

        case_name = csv_file.stem

        # ---------- T_pred ----------
        plot_3d_scatter(
            x, y, z, T_pred,
            title=f"{case_name}: $T_{{pred}}$",
            cbar_label="Temperature (K)",
            save_path=out_dir / "T_pred" / f"{case_name}_T_pred_3d.png",
            cmap="turbo",
            norm=plt.Normalize(vmin=T_min, vmax=T_max)
        )

        # ---------- Error ----------
        plot_3d_scatter(
            x, y, z, err,
            title=f"{case_name}: $T_{{true}} - T_{{pred}}$",
            cbar_label="Temperature Error (K)",
            save_path=out_dir / "error" / f"{case_name}_error_3d.png",
            cmap="coolwarm",
            norm=err_norm
        )

        print(f"[OK] Visualized {case_name}")

    print(f"[DONE] Results saved under: {out_dir}")


if __name__ == "__main__":
    main()

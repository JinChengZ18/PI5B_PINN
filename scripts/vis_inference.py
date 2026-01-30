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
from matplotlib.gridspec import GridSpec


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
    "font.family": "Times New Roman",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
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
    gs = GridSpec(1, 2, width_ratios=[1, 0.03], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    cax = fig.add_subplot(gs[0, 1])

    sc = ax.scatter(
        x, y, z,
        c=values,
        cmap=cmap,
        norm=norm,
        s=1,
        alpha=0.85,
        linewidth=0
    )

    ax.set_title(title, pad=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=20, azim=-135)
    ax.grid(False)

    cb = fig.colorbar(sc, cax=cax, shrink=0.50)
    cb.set_label(cbar_label)

    fig.subplots_adjust(
        left=0.04,
        right=0.96,
        top=0.90,
        bottom=0.06
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)



def plot_3d_scatter_pair(
    x, y, z,
    values_left, values_right,
    title_left, title_right,
    cbar_label,
    save_path,
    cmap,
    norm=None,
    max_points=50000
):
    if len(x) > max_points:
        idx = np.random.choice(len(x), max_points, replace=False)
        x, y, z = x[idx], y[idx], z[idx]
        values_left = values_left[idx]
        values_right = values_right[idx]

    fig = plt.figure(figsize=(15, 6))

    # -----------------------------
    # GridSpec: [left | right | cbar]
    # -----------------------------
    gs = GridSpec(
        1, 3,
        width_ratios=[1, 1, 0.03],
        wspace=0.12
    )

    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    cax = fig.add_subplot(gs[0, 2])

    sc1 = ax1.scatter(
        x, y, z,
        c=values_left,
        cmap=cmap,
        norm=norm,
        s=1,
        alpha=0.85,
        linewidth=0
    )

    sc2 = ax2.scatter(
        x, y, z,
        c=values_right,
        cmap=cmap,
        norm=norm,
        s=1,
        alpha=0.85,
        linewidth=0
    )

    # -----------------------------
    # Axes formatting
    # -----------------------------
    for ax, title in zip([ax1, ax2], [title_left, title_right]):
        ax.set_title(title, pad=12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=20, azim=-135)  # 论文级固定视角
        ax.grid(False)

    # -----------------------------
    # Colorbar (独立轴，绝不重叠)
    # -----------------------------
    cb = fig.colorbar(sc1, cax=cax, shrink=0.5)
    cb.set_label(cbar_label)

    # -----------------------------
    # 手动控制边距（替代 tight_layout）
    # -----------------------------
    fig.subplots_adjust(
        left=0.04,
        right=0.96,
        top=0.90,
        bottom=0.06
    )

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
        default="output/M02/thermal_heat_source",
        help="Inference CSV directory"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="vis/inference/M02",
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
        
        # ---------- T_pred & T_true ----------
        plot_3d_scatter_pair(
            x, y, z,
            T_pred, T_true,
            title_left=f"{case_name}: $T_{{pred}}$",
            title_right=f"{case_name}: $T_{{true}}$",
            cbar_label="Temperature (K)",
            save_path=out_dir / "T_pair" / f"{case_name}_T_pred_T_true_3d.png",
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

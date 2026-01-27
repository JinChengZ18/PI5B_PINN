#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ==========================================================
# 论文级绘图风格
# ==========================================================
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.dpi": 200,
    "savefig.dpi": 300,
})


# ==========================================================
# 可视化函数
# ==========================================================
def plot_field(
    x, y, values,
    title,
    cbar_label,
    save_path,
    cmap="viridis",
    norm=None
):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    sc = ax.scatter(
        x, y,
        c=values,
        s=6,
        cmap=cmap,
        norm=norm,
        linewidths=0
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


# ==========================================================
# 主逻辑
# ==========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visualize inference results: T_pred and T_true - T_pred"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="output/M02/thermal_heat_source",
        help="Inference output directory"
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

    # scripts/ 与 output/ 平级，回到项目根目录
    os.chdir(Path(__file__).resolve().parent.parent)

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    assert input_dir.exists(), f"Input directory not found: {input_dir}"

    csv_files = sorted(input_dir.glob("case_*.csv"))
    if args.max_cases is not None:
        csv_files = csv_files[:args.max_cases]

    print(f"[INFO] Found {len(csv_files)} cases")

    # ------------------------------------------------------
    # 先扫描所有 case，确定统一的色标范围
    # ------------------------------------------------------
    all_T_pred = []
    all_err = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        T_pred = df["T_pred"].values
        T_true = df["T_true"].values
        err = T_true - T_pred

        all_T_pred.append(T_pred)
        all_err.append(err)

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
    # 逐 case 可视化
    # ------------------------------------------------------
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        x = df["x"].values
        y = df["y"].values
        T_pred = df["T_pred"].values
        T_true = df["T_true"].values
        err = T_true - T_pred

        case_name = csv_file.stem

        # ---------- T_pred ----------
        plot_field(
            x, y, T_pred,
            title=f"{case_name}: $T_{{pred}}$",
            cbar_label="Temperature (K)",
            save_path=out_dir / "T_pred" / f"{case_name}_T_pred.png",
            cmap="inferno",
            norm=plt.Normalize(vmin=T_min, vmax=T_max)
        )

        # ---------- Error ----------
        plot_field(
            x, y, err,
            title=f"{case_name}: $T_{{true}} - T_{{pred}}$",
            cbar_label="Temperature Error (K)",
            save_path=out_dir / "error" / f"{case_name}_error.png",
            cmap="coolwarm",
            norm=err_norm
        )

        print(f"[OK] Visualized {case_name}")

    print(f"[DONE] Results saved under: {out_dir}")


if __name__ == "__main__":
    main()


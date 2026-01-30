#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference parity plot visualization
- T_pred vs T_true
- Point density coloring (KDE)
- Identity line y = x
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ==========================================================
# 项目根目录
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ==========================================================
# 论文级绘图风格（保持一致）
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
# Parity plot 函数
# ==========================================================
def plot_parity_density(
    T_true,
    T_pred,
    title,
    save_path,
    bins=300,
    max_points=200000
):
    if len(T_true) > max_points:
        idx = np.random.choice(len(T_true), max_points, replace=False)
        T_true = T_true[idx]
        T_pred = T_pred[idx]

    # --------------------------------------------------
    # 2D histogram density (FAST)
    # --------------------------------------------------
    H, xedges, yedges = np.histogram2d(
        T_true, T_pred, bins=bins
    )

    # Map each point to its bin density
    x_idx = np.searchsorted(xedges, T_true, side="right") - 1
    y_idx = np.searchsorted(yedges, T_pred, side="right") - 1

    valid = (
        (x_idx >= 0) & (x_idx < bins) &
        (y_idx >= 0) & (y_idx < bins)
    )

    density = np.zeros_like(T_true)
    density[valid] = H[x_idx[valid], y_idx[valid]]

    # log scale for visual separation
    density = np.log10(density + 1)

    # sort so dense points are on top
    order = density.argsort()
    T_true = T_true[order]
    T_pred = T_pred[order]
    density = density[order]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    sc = ax.scatter(
        T_true,
        T_pred,
        c=density,
        s=3,
        cmap="magma",   # 🔥 论文级
        alpha=0.9,
        rasterized=True  # PDF 友好 & 快
    )

    vmin = min(T_true.min(), T_pred.min())
    vmax = max(T_true.max(), T_pred.max())
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1)

    ax.set_xlabel(r"$T_{\mathrm{true}}$ (K)")
    ax.set_ylabel(r"$T_{\mathrm{pred}}$ (K)")
    ax.set_title(title)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

    cb = plt.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label(r"$\log_{10}(\mathrm{point\ density})$")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


# ==========================================================
# 主流程
# ==========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Parity plot: T_pred vs T_true with density coloring"
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
        default="vis/inference_parity/M02",
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

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        T_true = df["T_true"].values
        T_pred = df["T_pred"].values

        case_name = csv_file.stem

        plot_parity_density(
            T_true,
            T_pred,
            title=f"{case_name}: $T_{{pred}}$ vs $T_{{true}}$",
            save_path=out_dir / "parity" / f"{case_name}_parity.png"
        )

        print(f"[OK] Visualized {case_name}")

    print(f"[DONE] Results saved under: {out_dir}")

if __name__ == "__main__":
    main()

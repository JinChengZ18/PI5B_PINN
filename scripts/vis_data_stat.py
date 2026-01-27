"""
vis_data_stat.py

按参数分层聚合温度场，并生成参数变化 GIF
"""

# ==========================================================
# 强制使用 Agg 后端（无 GUI，稳定）
# ==========================================================
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ==========================================================
# 标准库
# ==========================================================
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio

# ==========================================================
# 项目路径
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data/thermal_heat_source"
INDEX_FILE = DATA_DIR / "index.jsonl"

STAT_DIR = PROJECT_ROOT / "data/thermal_heat_source_stat"
GIF_DIR = PROJECT_ROOT / "data/thermal_heat_source_stat_gif"
STAT_DIR.mkdir(exist_ok=True)
GIF_DIR.mkdir(exist_ok=True)

PARAM_KEYS = ["soc_power", "pmic_power", "usb_power", "other_power"]

# ==========================================================
# COMSOL CSV 读取
# ==========================================================
def read_comsol_csv(path):
    return pd.read_csv(
        path,
        comment="%",
        header=None,
        names=["x", "y", "z", "T (K)", "spf.U (m/s)"]
    )

# ==========================================================
# 读取 index.jsonl
# ==========================================================
def load_index():
    records = []
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rec["csv_path"] = Path(rec["export_file"])
            records.append(rec)
    return records

# ==========================================================
# 按参数分组
# ==========================================================
def group_by_parameter(records, param_key):
    groups = {}
    for r in records:
        val = r["parameters"][param_key]
        groups.setdefault(val, []).append(r)
    return dict(sorted(groups.items()))

# ==========================================================
# 聚合温度场（逐点平均）
# ==========================================================
def aggregate_temperature(records):
    dfs = [read_comsol_csv(r["csv_path"]) for r in records]

    base = dfs[0][["x", "y", "z"]].copy()
    temps = np.stack([df["T (K)"].values for df in dfs], axis=0)

    base["T (K)"] = temps.mean(axis=0)
    return base

# ==========================================================
# 绘制单帧
# ==========================================================
def render_frame(df, param_name, param_value, save_path, max_points=12000):
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)

    fig = plt.figure(figsize=(10, 8), dpi=200)
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        df["x"], df["y"], df["z"],
        c=df["T (K)"],
        cmap="turbo",
        s=2,
        vmin=290,
        vmax=500,
        linewidths=0
    )

    ax.set_title("参数分层平均温度场")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")

    cb = plt.colorbar(sc, ax=ax, shrink=0.6)
    cb.set_label("温度 (K)")

    ax.text2D(
        0.98, 0.98,
        f"变化参数: {param_name}\n当前值: {param_value}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8)
    )

    plt.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    return frame

# ==========================================================
# 主流程
# ==========================================================
def main():
    records = load_index()

    for param in PARAM_KEYS:
        print(f"\n=== 处理参数: {param} ===")

        param_dir = STAT_DIR / param
        param_dir.mkdir(exist_ok=True)

        groups = group_by_parameter(records, param)
        frames = []

        for val, group_records in groups.items():
            print(f"  聚合 {param} = {val} ({len(group_records)} cases)")

            agg_df = aggregate_temperature(group_records)

            csv_path = param_dir / f"{param}_{val}.csv"
            agg_df.to_csv(csv_path, index=False)

            frame = render_frame(
                agg_df,
                param_name=param,
                param_value=val,
                save_path=None,
                max_points=60000
            )
            frames.append(frame)

        gif_path = GIF_DIR / f"{param}.gif"
        imageio.mimsave(gif_path, frames, fps=2, subrectangles=True)
        print(f"  ✓ GIF 已生成: {gif_path}")

    print("\n所有参数分层统计与 GIF 生成完成！")

# ==========================================================
if __name__ == "__main__":
    main()

"""
3D 热场散点图可视化脚本（带参数标注）
逐文件绘制并保存 PNG
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# ==========================================================
# 添加项目根目录到路径（与 test_dataset 一致）
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 数据目录
DATA_DIR = PROJECT_ROOT / "data/thermal_heat_source"
INDEX_FILE = DATA_DIR / "index.jsonl"
OUT_DIR = PROJECT_ROOT / "data/thermal_heat_source_visualization"
OUT_DIR.mkdir(exist_ok=True)

# ==========================================================
# 统一的 COMSOL CSV 读取函数
# ==========================================================
def read_comsol_csv(path):
    return pd.read_csv(
        path,
        comment="%",
        header=None,
        names=["x", "y", "z", "T (K)", "spf.U (m/s)"]
    )


# ==========================================================
# 读取 index.jsonl → {case_id: parameters}
# ==========================================================
def load_case_parameters(index_file):
    case_params = {}
    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            case_params[record["case_id"]] = record["parameters"]
    return case_params


# ==========================================================
# 3D 散点图绘制函数（修复 text 错误）
# ==========================================================
def plot_3d_scatter(df, title, params, save_path, max_points=50000):

    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)

    x = df["x"].values
    y = df["y"].values
    z = df["z"].values
    t = df["T (K)"].values

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        x, y, z,
        c=t,
        cmap="turbo",
        s=1,
        alpha=0.8
    )

    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")

    cb = plt.colorbar(sc, ax=ax, shrink=0.6)
    cb.set_label("Temperature (K)")

    # ======================================================
    # 修复：使用 ax.text2D() 而不是 plt.text()
    # ======================================================
    param_text = (
        f"SoC: {params['soc_power']} W\n"
        f"PMIC: {params['pmic_power']} W\n"
        f"USB: {params['usb_power']} W\n"
        f"Other: {params['other_power']} W"
    )

    ax.text2D(
        0.98, 0.98, param_text,
        transform=ax.transAxes,
        fontsize=10,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


# ==========================================================
# 主流程：逐文件绘制
# ==========================================================
def main():
    csv_files = sorted(DATA_DIR.glob("case_*.csv"))
    if not csv_files:
        print("未找到 CSV 文件")
        return

    case_params = load_case_parameters(INDEX_FILE)

    print(f"找到 {len(csv_files)} 个 CSV 文件，开始绘制...")

    for csv_file in csv_files:
        case_id = csv_file.stem
        print(f"处理 {csv_file.name} ...")

        df = read_comsol_csv(csv_file)

        params = case_params.get(case_id)
        if params is None:
            print(f"⚠ 未找到 {case_id} 的参数，跳过")
            continue

        save_path = OUT_DIR / f"{case_id}_3d.png"

        plot_3d_scatter(
            df,
            title=f"3D Thermal Field - {case_id}",
            params=params,
            save_path=save_path
        )

    print(f"\n全部绘制完成！输出目录：{OUT_DIR}")


if __name__ == "__main__":
    main()

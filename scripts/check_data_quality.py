"""
PINN 数据自动质量检查脚本
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================================
# 添加项目根目录到 Python 路径（与 test_dataset 一致）
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ==========================================================
# 路径配置（基于项目根目录）
# ==========================================================
DATA_DIR = PROJECT_ROOT / "data/thermal_heat_source"
INDEX_FILE = DATA_DIR / "index.jsonl"

EXPECTED_COLUMNS = ["x", "y", "z", "T (K)", "spf.U (m/s)"]

COORD_RANGE = {
    "x": (0.0, 85.0),
    "y": (0.0, 56.0),
    "z": (0.0, 50.0),
}
TEMP_RANGE = (273.0, 400.0)  # K
MIN_CSV_SIZE_MB = 30.0


# ==========================================================
# 工具函数
# ==========================================================
def fail(msg):
    raise RuntimeError(f"[FAILED] {msg}")


def info(msg):
    print(f"[INFO] {msg}")


# ==========================================================
# 1. CSV 文件完整性
# ==========================================================
def check_csv_integrity():
    csv_files = sorted(DATA_DIR.glob("case_*.csv"))

    if not csv_files:
        fail("未找到任何 CSV 文件")

    info(f"发现 CSV 文件数量: {len(csv_files)}")

    for f in csv_files:
        size_mb = f.stat().st_size / 1024 / 1024
        if size_mb < MIN_CSV_SIZE_MB:
            fail(f"{f.name} 文件过小 ({size_mb:.2f} MB)")

    info("✓ CSV 文件完整性检查通过")
    return csv_files


# ==========================================================
# 2. 列名一致性
# ==========================================================
def check_column_consistency(csv_files):
    df = pd.read_csv(csv_files[0], comment="%", header=None, names=EXPECTED_COLUMNS)
    if list(df.columns) != EXPECTED_COLUMNS:
        fail(f"列名不一致: {df.columns.tolist()}")

    info("✓ CSV 列名一致性检查通过")


# ==========================================================
# 3. 数值范围有效性
# ==========================================================
def check_value_ranges(csv_files, sample_n=3):
    for f in csv_files[:sample_n]:
        df = pd.read_csv(f, comment="%", header=None, names=EXPECTED_COLUMNS)

        for axis, (vmin, vmax) in COORD_RANGE.items():
            if df[axis].min() < vmin or df[axis].max() > vmax:
                fail(f"{f.name} 中 {axis} 超出范围")

        if df["T (K)"].min() < TEMP_RANGE[0] or df["T (K)"].max() > TEMP_RANGE[1]:
            fail(f"{f.name} 中温度超出合理范围")

    info("✓ 数值范围检查通过")


# ==========================================================
# 4. NaN / Inf 检查
# ==========================================================
def check_nan_inf(csv_files, sample_n=3):
    for f in csv_files[:sample_n]:
        df = pd.read_csv(f, comment="%", header=None, names=EXPECTED_COLUMNS)
        if not np.isfinite(df[["x", "y", "z", "T (K)"]].values).all():
            fail(f"{f.name} 中存在 NaN 或 Inf")

    info("✓ NaN / Inf 检查通过")


# ==========================================================
# 5. 坐标网格均匀性
# ==========================================================
def check_grid_uniformity(csv_files):
    df = pd.read_csv( csv_files[0], comment="%", header=None, names=EXPECTED_COLUMNS)

    for axis in ["x", "y", "z"]:
        vals = np.sort(df[axis].unique())
        diffs = np.diff(vals)

        if len(diffs) == 0:
            fail(f"{axis} 方向坐标点不足")

        ratio = diffs.max() / diffs.min()
        if ratio > 1.5:
            fail(f"{axis} 网格不均匀 (max/min={ratio:.2f})")

    info("✓ 坐标网格均匀性检查通过")


# ==========================================================
# 6. index.jsonl 完整性
# ==========================================================
def check_index_file(csv_files):
    if not INDEX_FILE.exists():
        fail("index.jsonl 不存在")

    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    if len(records) != len(csv_files):
        fail("index.jsonl 行数与 CSV 文件数量不一致")

    required_keys = {"soc_power", "pmic_power", "usb_power", "other_power"}

    for r in records:
        if not Path(r["export_file"]).exists():
            fail(f"CSV 路径不存在: {r['export_file']}")

        if set(r["parameters"].keys()) != required_keys:
            fail(f"参数字段不完整: {r['parameters'].keys()}")

    info("✓ index.jsonl 完整性检查通过")


# ==========================================================
# 主入口
# ==========================================================
def main():
    print("=" * 60)
    print("PINN 数据自动质量检查")
    print("=" * 60)

    csv_files = check_csv_integrity()
    check_column_consistency(csv_files)
    check_value_ranges(csv_files)
    check_nan_inf(csv_files)
    check_grid_uniformity(csv_files)
    check_index_file(csv_files)

    print("=" * 60)
    print("✓ 所有数据质量检查通过")
    print("=" * 60)


if __name__ == "__main__":
    main()

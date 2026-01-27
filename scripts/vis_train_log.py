import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt


# =========================
# 全局绘图风格（学术化）
# =========================
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.0,
    "figure.dpi": 150,
})


EPOCH_LOSS_PATTERN = re.compile(
    r"\[Epoch\s+(\d+)\]\s+MSE Loss\s+=\s+([0-9.eE+-]+)"
)


def parse_log_file(log_path: Path):
    epochs = []
    losses = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = EPOCH_LOSS_PATTERN.search(line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))

    return epochs, losses


def plot_loss_curve(epochs, losses, save_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    ax.plot(
        epochs,
        losses,
        color="#1f77b4",      # 学术常用深蓝
        marker="o",
        markersize=3,
        markerfacecolor="white",
        markeredgewidth=0.8,
        label="Training Loss"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)

    # ✅ 对数坐标轴
    ax.set_yscale("log")

    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.legend(frameon=False)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="可视化训练日志中的 epoch-loss 曲线",
        usage="python scripts/vis_train_log.py --log_dir logs/M01/exp01"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/M01/exp01",
        help="日志目录，例如 logs/M01/exp01"
    )
    args = parser.parse_args()

    # scripts/ 与 logs/ 平级，先回到项目根目录
    os.chdir(Path(__file__).resolve().parent.parent)

    log_root = Path(args.log_dir)
    assert log_root.exists(), f"Log dir not found: {log_root}"

    for log_file in log_root.glob("*.log"):
        epochs, losses = parse_log_file(log_file)

        if not epochs:
            print(f"[WARN] No epoch-loss found in {log_file}")
            continue

        relative_path = log_file.relative_to("logs")
        save_path = Path("vis") / relative_path.with_suffix(".png")

        plot_loss_curve(
            epochs,
            losses,
            save_path,
            title=log_file.stem
        )

        print(f"[OK] Saved: {save_path}")


if __name__ == "__main__":
    main()

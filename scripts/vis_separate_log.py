# vis_separate_log.py
import argparse
import os
import re
from collections import defaultdict, OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 支持 MSE Loss 和 PDE Loss 的正则表达式
EPOCH_LOSS_PATTERN = re.compile(
    r"\[Epoch\s+(\d+)\]\s+MSE Loss\s*=\s*([0-9.eE+-]+)(?:\s*\|\s*PDE Loss\s*=\s*([0-9.eE+-]+))?"
)

MODEL_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#8c564b", "#e377c2", "#7f7f7f", "#17becf", "#bcbd22",
    "#1a55FF", "#B00020",
]

EXP_LINESTYLES = OrderedDict({
    "exp01": (0, ()),
    "exp02": (0, (6, 2)),
    "exp03": (0, (1, 1)),
    "exp04": (0, (4, 1, 1, 1)),
    "exp05": (0, (8, 2, 1, 2)),
    "exp06": (0, (3, 2, 1, 2)),
    "exp07": (0, (1, 3)),
    "exp08": (0, (10, 3)),
    "exp09": (0, (2, 2, 2, 2)),
})


def configure_plot_style(usetex: bool):
    plt.style.use("seaborn-v0_8-paper")

    if usetex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
        })
    else:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
        })

    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 19,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "lines.linewidth": 2.0,
        "figure.dpi": 200,
        "savefig.dpi": 300,
    })


def parse_log_file(log_path: Path):
    epochs, data_losses, pde_losses = [], [], []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = EPOCH_LOSS_PATTERN.search(line)
            if m:
                epochs.append(int(m.group(1)))
                data_losses.append(float(m.group(2)))
                # 如果有 PDE Loss，就添加；否则用 None 占位
                pde_losses.append(float(m.group(3)) if m.group(3) is not None else None)
    return epochs, data_losses, pde_losses


def find_all_logs(logs_root: Path):
    logs_by_model = defaultdict(lambda: defaultdict(list))
    for model_dir in sorted(logs_root.glob("M*")):
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for exp_dir in sorted(model_dir.glob("exp*")):
            if not exp_dir.is_dir():
                continue
            exp = exp_dir.name
            for log_file in sorted(exp_dir.glob("*.log")):
                logs_by_model[model][exp].append(log_file)
    return logs_by_model


def exp_to_linestyle(exp_name: str):
    if exp_name in EXP_LINESTYLES:
        return EXP_LINESTYLES[exp_name]
    m = re.match(r"exp(\d+)", exp_name)
    if m:
        idx = (int(m.group(1)) - 1) % len(EXP_LINESTYLES)
        return list(EXP_LINESTYLES.values())[idx]
    return (0, ())  # fallback solid


def plot_model_loss(
    model: str,
    logs_by_exp: dict,
    out_dir: Path,
    use_log_y: bool,
):
    fig, ax = plt.subplots(figsize=(12, 7))

    plotted_any = False
    for exp, log_files in sorted(logs_by_exp.items()):
        ls = exp_to_linestyle(exp)
        for k, log_file in enumerate(log_files):
            epochs, data_losses, pde_losses = parse_log_file(log_file)
            if not epochs:
                continue
            plotted_any = True
            alpha = max(0.35, 0.85 - 0.15 * k)
            label = exp if k == 0 else None

            # 绘制 Data Loss
            ax.plot(
                epochs,
                data_losses,
                linestyle=ls,
                alpha=alpha,
                label=f"{label} - Data",
                color=MODEL_COLORS[0],
            )

            # 如果 PDE Loss 存在，绘制 PDE Loss
            if any(pde is not None for pde in pde_losses):
                ax.plot(
                    epochs,
                    [pde if pde is not None else float("nan") for pde in pde_losses],
                    linestyle=ls,
                    alpha=alpha,
                    label=f"{label} - PDE",
                    color=MODEL_COLORS[1],
                )

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"{model} Training Loss")

    if use_log_y:
        ax.set_yscale("log")

    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{model}_loss.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="将不同模型的训练loss曲线拆开绘图，每个模型一张图。"
    )
    parser.add_argument("--logs_root", type=str, default="logs")
    parser.add_argument("--out_dir", type=str, default="vis/separate_models")
    parser.add_argument("--usetex", action="store_true")
    parser.add_argument("--linear", action="store_true")

    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)

    # ✅ 样式只在这里设置一次，且在任何 figure 创建之前
    configure_plot_style(args.usetex)

    logs_by_model = find_all_logs(Path(args.logs_root))
    use_log_y = not args.linear

    for model, logs_by_exp in logs_by_model.items():
        plot_model_loss(
            model=model,
            logs_by_exp=logs_by_exp,
            out_dir=Path(args.out_dir),
            use_log_y=use_log_y,
        )


if __name__ == "__main__":
    main()

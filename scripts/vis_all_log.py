#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from collections import defaultdict, OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


EPOCH_LOSS_PATTERN = re.compile(
    r"\[Epoch\s+(\d+)\]\s+MSE Loss\s+=\s+([0-9.eE+-]+)"
)

# 高区分度（论文常用）颜色池：颜色=模型
MODEL_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#17becf",  # cyan
    "#bcbd22",  # olive
    "#1a55FF",  # vivid blue
    "#B00020",  # deep red
]

# 线型=同一模型内不同实验 expxx（支持 9 个，超出会循环/退化为 solid）
# Matplotlib dash pattern: (offset, (on_off_seq...))
EXP_LINESTYLES = OrderedDict({
    "exp01": (0, ()),                  # solid
    "exp02": (0, (6, 2)),              # dashed
    "exp03": (0, (1, 1)),              # dotted
    "exp04": (0, (4, 1, 1, 1)),        # dash-dot
    "exp05": (0, (8, 2, 1, 2)),        # long dash-dot
    "exp06": (0, (3, 2, 1, 2)),        # custom
    "exp07": (0, (1, 3)),              # sparse dots
    "exp08": (0, (10, 3)),             # very long dash
    "exp09": (0, (2, 2, 2, 2)),        # even dash
})


def configure_plot_style(usetex: bool):
    plt.style.use("seaborn-v0_8-paper")

    # 字体与 LaTeX
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

    # 论文级默认参数
    plt.rcParams.update({
    # 全局基础字号
    "font.size": 16,
    # 坐标轴
    "axes.labelsize": 19,
    "axes.titlesize": 20,
    # 刻度
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    # 图例
    "legend.fontsize": 16,
    # 线条与分辨率
    "lines.linewidth": 2.0,
    "figure.dpi": 200,
    "savefig.dpi": 300,
})



def parse_log_file(log_path: Path):
    epochs, losses = [], []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = EPOCH_LOSS_PATTERN.search(line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return epochs, losses


def find_all_logs(logs_root: Path):
    """
    扫描目录：
      logs_root/M*/exp*/*.log

    返回：
      logs_by_model[model][exp] = [log_files...]
    """
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
    # 如果 exp01-exp09 显式定义，就用定义；否则按编号循环映射
    if exp_name in EXP_LINESTYLES:
        return EXP_LINESTYLES[exp_name]

    # 尝试提取 exp 的数字部分，做稳定映射
    m = re.match(r"exp(\d+)", exp_name)
    if m:
        idx = (int(m.group(1)) - 1) % len(EXP_LINESTYLES)
        return list(EXP_LINESTYLES.values())[idx]

    return (0, ())  # fallback solid


def plot_all_models(
    logs_by_model,
    out_path: Path,
    title: str,
    use_log_y: bool,
    legend_mode: str,
    max_exp_legend: int | None,
):
    # 大图：信息多时必须给足画布
    fig, ax = plt.subplots(figsize=(20, 11))

    models = sorted(logs_by_model.keys())
    model_color = {
        model: MODEL_COLORS[i % len(MODEL_COLORS)]
        for i, model in enumerate(models)
    }

    # 用于 exp legend：只收集实际出现过的 exp
    seen_exps = set()
    plotted_any = False

    for model in models:
        color = model_color[model]
        exps = sorted(logs_by_model[model].keys())

        for exp in exps:
            seen_exps.add(exp)
            ls = exp_to_linestyle(exp)

            log_files = logs_by_model[model][exp]
            if not log_files:
                continue

            # 同一 exp 下可能多次 run：同色同线型，靠 alpha 区分
            for k, log_file in enumerate(log_files):
                epochs, losses = parse_log_file(log_file)
                if not epochs:
                    continue

                plotted_any = True

                # 同一 exp 的多条曲线，alpha 递减，避免完全重叠看不清
                alpha = max(0.35, 0.85 - 0.15 * k)

                if legend_mode == "all":
                    label = f"{model}/{exp}"
                else:
                    # legend 只按模型：每个模型仅标一次
                    label = model if (exp == exps[0] and k == 0) else None

                ax.plot(
                    epochs,
                    losses,
                    color=color,
                    linestyle=ls,
                    alpha=alpha,
                    label=label,
                )

    if not plotted_any:
        raise RuntimeError("No epoch-loss data found. Please check log format or directory paths.")

    ax.set_xlabel(r"Epoch")
    ax.set_ylabel(r"MSE loss")
    ax.set_title(title)

    # ✅ 默认 log-scale（论文级）；需要线性时用户用 --linear
    if use_log_y:
        ax.set_yscale("log")

    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)

    # ===== Legend 设计：必须让审稿人一眼看懂你的意图 =====
    if legend_mode == "models":
        # 1) 颜色=模型
        handles_models = [
            Line2D([0], [0], color=model_color[m], lw=3, label=m)
            for m in models
        ]
        leg1 = ax.legend(
            handles=handles_models,
            title=r"Model (color)",
            loc="upper right",
            frameon=False,
        )
        ax.add_artist(leg1)

        # 2) 线型=同一模型内不同实验 expxx
        exps_sorted = sorted(seen_exps)
        if max_exp_legend is not None:
            exps_sorted = exps_sorted[:max_exp_legend]

        handles_exp = [
            Line2D(
                [0], [0],
                color="black",
                lw=2,
                linestyle=exp_to_linestyle(exp),
                label=exp
            )
            for exp in exps_sorted
        ]

        ax.legend(
            handles=handles_exp,
            title=r"Experiment within each model (linestyle)",
            loc="upper center",
            ncol=min(5, max(1, len(handles_exp))),
            frameon=False,
        )
    else:
        # legend 全开：会很拥挤，但按需求提供
        ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="多模型/多实验训练日志loss对比：颜色=模型，线型=实验，默认y轴log-scale。",
    )
    parser.add_argument("--logs_root", type=str, default="logs", help="日志根目录，默认 logs/")
    parser.add_argument("--out", type=str, default="vis/all_models_loss.png", help="输出图片路径")
    parser.add_argument("--title", type=str, default="Training loss across models and experiments", help="图标题")
    parser.add_argument("--usetex", action="store_true", help="启用 LaTeX 字体渲染（需要本机安装LaTeX）")
    parser.add_argument(
        "--legend_mode",
        type=str,
        choices=["models", "all"],
        default="models",
        help="models: legend按模型+实验语义拆分（推荐）；all: 直接显示每条曲线label（易拥挤）",
    )
    parser.add_argument(
        "--max_exp_legend",
        type=int,
        default=9,
        help="实验线型legend最多展示多少个exp（默认9，超出会截断；设为0表示不显示实验legend）",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="使用线性 y 轴（默认使用 log-scale）",
    )

    args = parser.parse_args()

    # scripts/ 与 logs/ 平级：回项目根目录
    os.chdir(Path(__file__).resolve().parent.parent)

    configure_plot_style(args.usetex)

    logs_root = Path(args.logs_root)
    if not logs_root.exists():
        raise FileNotFoundError(f"logs_root not found: {logs_root}")

    logs_by_model = find_all_logs(logs_root)

    max_exp_legend = args.max_exp_legend
    if max_exp_legend is not None and max_exp_legend <= 0:
        max_exp_legend = None  # 不显示实验legend

    plot_all_models(
        logs_by_model=logs_by_model,
        out_path=Path(args.out),
        title=args.title,
        use_log_y=(not args.linear),
        legend_mode=args.legend_mode,
        max_exp_legend=max_exp_legend,
    )

    print(f"[OK] Saved: {args.out}")


if __name__ == "__main__":
    main()

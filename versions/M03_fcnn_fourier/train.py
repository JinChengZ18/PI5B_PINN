import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ==========================================================
# 项目根目录
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from M02_fcnn_anneal.dataset import ThermalHeatSourceDataset
from M02_fcnn_anneal.model import SimpleFourier
from M02_fcnn_anneal.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="FCNN-ANNEAL Training Script")

    # ---------- 训练超参数 ----------
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch_cases", type=int, default=10)
    parser.add_argument("--points_per_case", type=int, default=60000)
    parser.add_argument("--num_workers", type=int, default=16)

    # ---------- 运行配置 ----------
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_dir", type=str, default="logs/M02")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/M02")

    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # ================= 日志 =================
    log_dir = PROJECT_ROOT / args.log_dir
    logger = setup_logger(log_dir, name="train")

    logger.info("========== FCNN-Fourier 训练开始 ==========")
    logger.info(f"使用设备: {device}")
    logger.info(f"训练参数: {vars(args)}")

    # ================= 数据 =================
    dataset = ThermalHeatSourceDataset(
        points_per_case=args.points_per_case
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_cases,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logger.info(f"数据集大小: {len(dataset)} cases")
    logger.info(f"每 case 采样点数: {args.points_per_case}")

    # ================= 模型 =================
    model = SimpleFourier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ReduceLROnPlateau 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        min_lr=1e-5
    )

    logger.info(f"模型结构:\n{model}")
    logger.info(f"优化器: AdamW, 初始 lr={args.lr}, 调度策略: ReduceLROnPlateau(factor=0.5, patience=10)")

    # ================= 训练 =================
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for x7, T in loader:
            x7 = x7.squeeze(0).to(device)
            T = T.squeeze(0).to(device)

            pred = model(x7)
            loss = F.mse_loss(pred, T)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(f"[Epoch {epoch:03d}] MSE Loss = {avg_loss:.6f} | LR = {current_lr:.2e}")

        # 调度器根据 loss 自动调整学习率
        scheduler.step(avg_loss)

    # ================= 保存 =================
    ckpt_dir = PROJECT_ROOT / args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "M02_fcnn_anneal.pt"
    torch.save(model.state_dict(), ckpt_path)

    logger.info(f"模型已保存至: {ckpt_path}")
    logger.info("========== 训练结束 ==========")


if __name__ == "__main__":
    main()

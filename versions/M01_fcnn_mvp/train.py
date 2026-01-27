# M01fcnn_mvp/train.py
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

from M01_fcnn_mvp.dataset import ThermalHeatSourceDataset
from M01_fcnn_mvp.model import SimplePINN
from M01_fcnn_mvp.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="FCNN-MVP Training Script")

    # ---------- 训练超参数 ----------
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_cases", type=int, default=1)
    parser.add_argument("--points_per_case", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=4)

    # ---------- 运行配置 ----------
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_dir", type=str, default="logs/M01_fcnn_mvp")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")

    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # ================= 日志 =================
    log_dir = PROJECT_ROOT / args.log_dir
    logger = setup_logger(log_dir, name="train")

    logger.info("========== FCNN-MVP 训练开始 ==========")
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
    model = SimplePINN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    logger.info(f"模型结构:\n{model}")
    logger.info(f"优化器: AdamW, lr={args.lr}")

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
        logger.info(f"[Epoch {epoch:03d}] MSE Loss = {avg_loss:.6f}")

    # ================= 保存 =================
    ckpt_dir = PROJECT_ROOT / args.ckpt_dir
    ckpt_dir.mkdir(exist_ok=True)

    ckpt_path = ckpt_dir / "M01_fcnn_mvp.pt"
    torch.save(model.state_dict(), ckpt_path)

    logger.info(f"模型已保存至: {ckpt_path}")
    logger.info("========== 训练结束 ==========")


if __name__ == "__main__":
    main()

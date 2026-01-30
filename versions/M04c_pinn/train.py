# train.py
import sys
import argparse
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ==========================================================
# 项目根目录
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from M04c_pinn.dataset import ThermalHeatSourceDataset
from M04c_pinn.model import SimplePINN
from M04c_pinn.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="FCNN-ANNEAL Training Script")

    # ---------- 训练超参数 ----------
    parser.add_argument("--lambda_pde", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch_cases", type=int, default=1)

    # ===== 新增：PDE 高级功能（默认开启）=====
    parser.add_argument("--use_pde_anneal", action="store_true", default=False)
    parser.add_argument("--pde_anneal_gamma", type=float, default=0.03)
    parser.add_argument("--pde_lambda_min", type=float, default=0.0)

    parser.add_argument("--use_soft_mask", action="store_true", default=False)
    parser.add_argument("--mask_alpha", type=float, default=40.0)
    parser.add_argument("--mask_temp_ratio", type=float, default=0.7)

    # ---------- 显存感知 points_per_case ----------
    free_mem = torch.cuda.mem_get_info()[0] / 1024**2  # MiB
    if free_mem < 1000:
        parser.add_argument("--points_per_case", type=int, default=8000)
    elif free_mem < 5000:
        parser.add_argument("--points_per_case", type=int, default=40000)
    elif free_mem < 10000:
        parser.add_argument("--points_per_case", type=int, default=80000)
    elif free_mem < 20000:
        parser.add_argument("--points_per_case", type=int, default=160000)
    else:
        parser.add_argument("--points_per_case", type=int, default=320000)

    # ---------- 运行配置 ----------
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--log_dir", type=str, default="logs/M04c")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/M04c")

    return parser.parse_args()


def compute_lambda_pde(epoch, args):
    if not args.use_pde_anneal:
        return args.lambda_pde
    lam = args.lambda_pde * math.exp(-args.pde_anneal_gamma * (epoch - 1))
    return max(args.pde_lambda_min, lam)


def compute_soft_mask(T, args):
    T_max = T.max()
    thresh = args.mask_temp_ratio * T_max
    if args.use_soft_mask:
        return torch.sigmoid(args.mask_alpha * (thresh - T))
    else:
        return (T < thresh).float()




def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # ================= 日志 =================
    log_dir = PROJECT_ROOT / args.log_dir
    logger = setup_logger(log_dir, name="train")
    logger.info("========== PINN 训练开始 ==========")
    logger.info(f"物理项权重 λ_pde = {args.lambda_pde}")
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

    # ReduceLROnPlateau 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=6,
        min_lr=1e-5
    )

    logger.info(f"模型结构:\n{model}")
    logger.info(f"优化器: AdamW, 初始 lr={args.lr}, 调度策略: ReduceLROnPlateau(factor=0.5, patience=6)")

    # ================= 训练 =================
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_loss_pde = 0.0, 0.0

        lambda_pde_t = compute_lambda_pde(epoch, args)

        for x7, T in loader:
            try:
                x7 = x7.squeeze(0).to(device)
                T = T.squeeze(0).to(device)
                x7.requires_grad_(True)

                pred = model(x7)
                loss_data = F.mse_loss(pred, T)

                lap = model.laplacian(x7)
                pde_weight = compute_soft_mask(T, args)
                loss_pde = torch.mean((lap ** 2) * pde_weight)

                loss = loss_data + lambda_pde_t * loss_pde

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_loss_pde += loss_pde.item()

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning(f"[OOM] Epoch {epoch}: 降低采样点数重试")
                args.points_per_case = max(10000, args.points_per_case // 2)
                dataset = ThermalHeatSourceDataset(points_per_case=args.points_per_case)
                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_cases,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                loss_pde = torch.tensor(0.0)
                break

        avg_loss = total_loss / len(loader)
        avg_loss_pde = total_loss_pde / len(loader)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"[Epoch {epoch:03d}] "
            f"MSE Loss={avg_loss:.6f} | "
            f"PDE Loss={avg_loss_pde:.6f} | "
            f"λ_pde={lambda_pde_t:.3e} | "
            f"LR={current_lr:.2e}"
        )

        # 调度器根据 loss 自动调整学习率
        scheduler.step(avg_loss)

    # ================= 保存 =================
    ckpt_dir = PROJECT_ROOT / args.ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "M04c_pinn.pt"
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"训练结束，模型已保存至: {ckpt_path}")
    logger.info("========== 训练结束 ==========")


if __name__ == "__main__":
    main()

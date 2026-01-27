# inference.py
import sys
import argparse
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader

# ==========================================================
# 项目根目录
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from M04_pinn.dataset import ThermalHeatSourceDataset
from M04_pinn.model import SimplePINN


def parse_args():
    parser = argparse.ArgumentParser(description="FCNN Inference Script")

    # 给默认值，避免在 IDE 里直接运行时报 args 缺失（SystemExit: 2）
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/M04/M04_pinn.pt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="M04",
        help="Model version name, e.g. M01, M02, M03b"
    )

    parser.add_argument("--points_per_case", type=int, default=60000)
    parser.add_argument("--batch_cases", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # ================= 数据 =================
    dataset = ThermalHeatSourceDataset(points_per_case=args.points_per_case)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_cases,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # ================= 模型 =================
    model = SimplePINN().to(device)
    ckpt_path = PROJECT_ROOT / args.ckpt_path
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # ================= 输出目录 =================
    output_root = PROJECT_ROOT / "output" / args.model_name / "thermal_heat_source"
    output_root.mkdir(parents=True, exist_ok=True)

    # ================= 推理 =================
    with torch.no_grad():
        for case_idx, batch in enumerate(loader):
            (x7, T_true) = batch[:2]
            # 兼容 batch_cases=1 的形状：[1, N, 7] -> [N, 7]
            # 若未来 batch_cases>1，这里会直接报错提醒（避免静默写错文件）
            if x7.ndim == 3 and x7.shape[0] == 1:
                x7 = x7.squeeze(0)
                T_true = T_true.squeeze(0)

            x7 = x7.to(device)
            T_true = T_true.to(device)

            T_pred = model(x7)

            x7_np = x7.cpu().numpy()
            T_true_np = T_true.cpu().numpy()
            T_pred_np = T_pred.cpu().numpy()

            records = []
            for i in range(x7_np.shape[0]):
                records.append({
                    "x": float(x7_np[i, 0]),
                    "y": float(x7_np[i, 1]),
                    "z": float(x7_np[i, 2]),
                    "T_true": float(T_true_np[i, 0]),
                    "T_pred": float(T_pred_np[i, 0]),
                })

            df = pd.DataFrame.from_records(records)

            case_name = f"case_{case_idx:04d}.csv"
            output_path = output_root / case_name
            df.to_csv(output_path, index=False)

    print(f"Inference results saved under: {output_root}")
    print(f"Loaded checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

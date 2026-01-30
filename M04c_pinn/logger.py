# logger.py
import logging
from pathlib import Path
from datetime import datetime
import os


def setup_logger(
    log_dir: Path,
    name: str = "train",
    level: int = logging.INFO,
):
    """
    创建统一日志器（并发安全）：
    - 时间戳 + PID，避免并发冲突
    - 同时输出到控制台和文件
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")
    pid = os.getpid()
    log_file = log_dir / f"{name}_{timestamp}_gpu{gpu}_pid{pid}.log"

    logger = logging.getLogger(f"{name}_{pid}")
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ---------- 文件日志 ----------
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(level)

    # ---------- 控制台日志 ----------
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(level)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"日志文件创建于: {log_file}")
    logger.info(f"进程 PID: {pid}")

    return logger

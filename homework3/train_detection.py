import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb

from homework.models import Detector, save_model
from homework.metrics import DetectionMetric
from homework.datasets.road_dataset import load_data


def train(
    exp_dir: str = "logs",
    num_epoch: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    log_dir = Path(exp_dir) / f"detector_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = Detector().to(device)

    train_data = load_data("drive_data/train", batch_size=batch_size, shuffle=True)
    val_data = load_data("drive_data/val", batch_size=batch_size, shuffle=False)

    seg_loss_func = torch.nn.CrossEntropyLoss()
    depth_loss_func = torch.nn.L1Loss()   # mean absolute error
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    val_metric = DetectionMetric()

    global_step = 0
    for epoch in range(num_epoch):
        model.train()

        for batch in train_data:
            img = batch["image"].to(device)
            track = batch["track"].to(device)      # (B, h, w) long labels
            depth = batch["depth"].to(device)       # (B, h, w) float in [0,1]

            logits, raw_depth = model(img)

            seg_loss = seg_loss_func(logits, track)
            depth_loss = depth_loss_func(torch.sigmoid(raw_depth), depth)
            loss = seg_loss + depth_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.add_scalar("train_loss", loss.item(), global_step)
            logger.add_scalar("train_seg_loss", seg_loss.item(), global_step)
            logger.add_scalar("train_depth_loss", depth_loss.item(), global_step)
            global_step += 1

        model.eval()
        val_metric.reset()
        with torch.inference_mode():
            for batch in val_data:
                img = batch["image"].to(device)
                track = batch["track"].to(device)
                depth = batch["depth"].to(device)

                pred, pred_depth = model.predict(img)
                val_metric.add(pred, track, pred_depth, depth)

        metrics = val_metric.compute()
        logger.add_scalar("val_iou", metrics["iou"], global_step)
        logger.add_scalar("val_depth_error", metrics["abs_depth_error"], global_step)

        if epoch % 5 == 0 or epoch == num_epoch - 1:
            print(
                f"Epoch {epoch+1}/{num_epoch}: "
                f"iou={metrics['iou']:.4f} "
                f"depth_err={metrics['abs_depth_error']:.4f} "
                f"tp_depth_err={metrics['tp_depth_error']:.4f}"
            )

    save_model(model)
    torch.save(model.state_dict(), log_dir / "detector.th")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)
    train(**vars(parser.parse_args()))

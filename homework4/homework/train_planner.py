import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb

from homework.models import load_model, save_model
from homework.metrics import PlannerMetric
from homework.datasets.road_dataset import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 2024,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name).to(device)

    pipeline = "default" if model_name == "cnn_planner" else "state_only"

    train_data = load_data("drive_data/train", transform_pipeline=pipeline, batch_size=batch_size, shuffle=True)
    val_data = load_data("drive_data/val", transform_pipeline=pipeline, batch_size=batch_size, shuffle=False)

    loss_func = torch.nn.SmoothL1Loss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    val_metric = PlannerMetric()

    best_val_error = float("inf")
    global_step = 0
    for epoch in range(num_epoch):
        model.train()

        for batch in train_data:
            batch = {k: v.to(device) for k, v in batch.items()}

            preds = model(**batch)
            labels = batch["waypoints"]
            mask = batch["waypoints_mask"]

            loss = loss_func(preds, labels)
            # weight lateral (index 1) more heavily than longitudinal (index 0)
            dim_weights = torch.tensor([1.0, 5.0], device=preds.device)
            loss = loss * dim_weights
            loss = (loss * mask[..., None]).sum() / mask.sum().clamp(min=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.add_scalar("train_loss", loss.item(), global_step)
            global_step += 1

        model.eval()
        val_metric.reset()
        with torch.inference_mode():
            for batch in val_data:
                batch = {k: v.to(device) for k, v in batch.items()}
                preds = model(**batch)
                val_metric.add(preds, batch["waypoints"], batch["waypoints_mask"])

        metrics = val_metric.compute()
        logger.add_scalar("val_longitudinal_error", metrics["longitudinal_error"], global_step)
        logger.add_scalar("val_lateral_error", metrics["lateral_error"], global_step)

        if epoch % 5 == 0 or epoch == num_epoch - 1:
            print(
                f"Epoch {epoch+1}/{num_epoch}: "
                f"long_err={metrics['longitudinal_error']:.4f} "
                f"lat_err={metrics['lateral_error']:.4f}"
            )

        # save whenever this epoch's combined error improves on the best seen so far
        combined_error = metrics["longitudinal_error"] + metrics["lateral_error"]
        if combined_error < best_val_error:
            best_val_error = combined_error
            save_model(model)
            torch.save(model.state_dict(), log_dir / f"{model_name}.th")
            print(f"  -> new best (combined_error={combined_error:.4f}), saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2024)
    train(**vars(parser.parse_args()))

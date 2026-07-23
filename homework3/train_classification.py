import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb

from homework.models import Classifier, save_model
from homework.metrics import AccuracyMetric
from homework.datasets.classification_dataset import load_data


def train(
    exp_dir: str = "logs",
    num_epoch: int = 30,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    log_dir = Path(exp_dir) / f"classifier_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = Classifier().to(device)

    train_data = load_data(
        "classification_data/train", transform_pipeline="aug",
        batch_size=batch_size, shuffle=True,
    )
    val_data = load_data("classification_data/val", transform_pipeline="default")

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_metric = AccuracyMetric()
    val_metric = AccuracyMetric()

    global_step = 0
    for epoch in range(num_epoch):
        model.train()
        train_metric.reset()

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            logits = model(img)
            loss = loss_func(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_metric.add(logits.argmax(dim=1), label)
            logger.add_scalar("train_loss", loss.item(), global_step)
            global_step += 1

        model.eval()
        val_metric.reset()
        with torch.inference_mode():
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                pred = model.predict(img)
                val_metric.add(pred, label)

        train_acc = train_metric.compute()["accuracy"]
        val_acc = val_metric.compute()["accuracy"]
        logger.add_scalar("train_accuracy", train_acc, global_step)
        logger.add_scalar("val_accuracy", val_acc, global_step)

        if epoch % 5 == 0 or epoch == num_epoch - 1:
            print(f"Epoch {epoch+1}/{num_epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    save_model(model)
    torch.save(model.state_dict(), log_dir / "classifier.th")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    train(**vars(parser.parse_args()))
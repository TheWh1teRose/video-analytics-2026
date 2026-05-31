from dataset import UCF101
from model import resnet, resnet_flow
from torch.utils.data import DataLoader
import os
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import wandb

NUM_EPOCHS = 50
NUM_WORKERS = 4
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 4


def train_video_consensus(model, x, y, B, K, optimizer, scaler, use_amp):
    """One video (K segments) per forward/backward to cap peak GPU memory."""
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for b in range(B):
        x_b = x[b * K : (b + 1) * K]
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits_b = model(x_b)
            loss_b = F.cross_entropy(logits_b.mean(dim=0, keepdim=True), y[b : b + 1])
        scaler.scale(loss_b / B).backward()
        loss_sum += loss_b.item()
    scaler.step(optimizer)
    scaler.update()
    return loss_sum / B


@torch.no_grad()
def video_consensus_predict(model, x, B, K, use_amp):
    logits = []
    for b in range(B):
        x_b = x[b * K : (b + 1) * K]
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits.append(model(x_b).mean(dim=0))
    return torch.stack(logits)


def make_loader(dataset, batch_size, shuffle):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "pin_memory": torch.cuda.is_available(),
    }
    if NUM_WORKERS > 0:
        kwargs["num_workers"] = NUM_WORKERS
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def class_comparison_table(rgb_correct, flow_correct, totals, classnames):
    rows = []
    for i, name in enumerate(classnames):
        if totals[i] == 0:
            continue
        rgb_acc = rgb_correct[i] / totals[i]
        flow_acc = flow_correct[i] / totals[i]
        rows.append([name, rgb_acc, flow_acc, int(totals[i]), rgb_acc - flow_acc])

    table = wandb.Table(
        columns=["class", "rgb_accuracy", "flow_accuracy", "count", "rgb_minus_flow"],
        data=rows,
    )
    return table


def main():
    try:
        mp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = UCF101(
        "data/data/train.txt", "data/data/mini_UCF", "data/data/mini_UCF_flow"
    )
    train_loader = make_loader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    eval_dataset = UCF101(
        "data/data/validation.txt",
        "data/data/mini_UCF",
        "data/data/mini_UCF_flow",
        validation=True,
    )
    eval_loader = make_loader(eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    model = resnet(len(train_dataset.classnames)).to(device)
    temporal_model = resnet_flow(len(train_dataset.classnames)).to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)

    optimizer_flow = optim.SGD(temporal_model.parameters(), lr=5e-3, momentum=0.9)
    scheduler_flow = optim.lr_scheduler.MultiStepLR(
        optimizer_flow,
        milestones=[12000, 18000],
        gamma=0.1,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    wandb.init(
        project="TSN",
        config={
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": 1e-3,
            "lr_flow": 5e-3,
            "num_classes": len(train_dataset.classnames),
            "device": str(device),
            "num_workers": NUM_WORKERS,
        },
    )

    def evaluate():
        model.eval()
        temporal_model.eval()
        total_loss = 0.0
        total_loss_flow = 0.0
        correct = 0
        correct_flow = 0
        total = 0
        num_batches = 0
        num_classes = len(train_dataset.classnames)
        rgb_correct_per_class = torch.zeros(num_classes)
        flow_correct_per_class = torch.zeros(num_classes)
        total_per_class = torch.zeros(num_classes)

        with torch.no_grad():
            for (x, flows), y in eval_loader:
                x = x.to(device, non_blocking=True)
                flows = flows.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                B, K = x.shape[:2]
                x = x.view(B * K, *x.shape[2:])

                B_flow, K_flow = flows.shape[:2]
                flows = flows.view(B_flow * K_flow, *flows.shape[2:])

                consensus = video_consensus_predict(model, x, B, K, use_amp)
                consensus_flow = video_consensus_predict(
                    temporal_model, flows, B_flow, K_flow, use_amp
                )

                loss = F.cross_entropy(consensus, y)
                loss_flow = F.cross_entropy(consensus_flow, y)

                pred_rgb = consensus.argmax(dim=1)
                pred_flow = consensus_flow.argmax(dim=1)

                total_loss += loss.item()
                total_loss_flow += loss_flow.item()
                correct += (pred_rgb == y).sum().item()
                correct_flow += (pred_flow == y).sum().item()
                total += y.size(0)
                num_batches += 1

                for c in range(num_classes):
                    mask = y == c
                    n = mask.sum().item()
                    if n == 0:
                        continue
                    total_per_class[c] += n
                    rgb_correct_per_class[c] += (pred_rgb[mask] == c).sum().item()
                    flow_correct_per_class[c] += (pred_flow[mask] == c).sum().item()

        class_table = class_comparison_table(
            rgb_correct_per_class,
            flow_correct_per_class,
            total_per_class,
            train_dataset.classnames,
        )

        return {
            "eval/loss": total_loss / num_batches,
            "eval/loss_flow": total_loss_flow / num_batches,
            "eval/accuracy": correct / total,
            "eval/accuracy_flow": correct_flow / total,
            "eval/class_comparison": class_table,
        }

    for epoch in range(NUM_EPOCHS):
        model.train()
        temporal_model.train()
        epoch_loss = 0.0
        epoch_loss_flow = 0.0
        num_batches = 0

        for (x, flows), y in train_loader:
            x = x.to(device, non_blocking=True)
            flows = flows.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            B, K = x.shape[:2]
            x = x.view(B * K, *x.shape[2:])

            B_flow, K_flow = flows.shape[:2]
            flows = flows.view(B_flow * K_flow, *flows.shape[2:])

            loss = train_video_consensus(
                model, x, y, B, K, optimizer, scaler, use_amp
            )
            del x
            if device.type == "cuda":
                torch.cuda.empty_cache()

            loss_flow = train_video_consensus(
                temporal_model,
                flows,
                y,
                B_flow,
                K_flow,
                optimizer_flow,
                scaler,
                use_amp,
            )

            print(loss_flow)

            epoch_loss += loss
            epoch_loss_flow += loss_flow
            num_batches += 1

        eval_metrics = evaluate()
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": epoch_loss / num_batches,
                "train/loss_flow": epoch_loss_flow / num_batches,
                **eval_metrics,
            }
        )

    wandb.finish()


if __name__ == "__main__":
    main()

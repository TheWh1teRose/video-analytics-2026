from dataset import UCF101
from model import resnet, resnet_flow
from torch.utils.data import DataLoader
import os
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import wandb

NUM_EPOCHS = 10
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "16"))


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
    chart_rows = []
    for i, name in enumerate(classnames):
        if totals[i] == 0:
            continue
        rgb_acc = rgb_correct[i] / totals[i]
        flow_acc = flow_correct[i] / totals[i]
        rows.append([name, rgb_acc, flow_acc, int(totals[i]), rgb_acc - flow_acc])
        chart_rows.append([name, "RGB", rgb_acc])
        chart_rows.append([name, "Flow", flow_acc])

    table = wandb.Table(
        columns=["class", "rgb_accuracy", "flow_accuracy", "count", "rgb_minus_flow"],
        data=rows,
    )
    chart_table = wandb.Table(
        columns=["class", "model", "accuracy"],
        data=chart_rows,
    )
    chart = wandb.plot.bar(
        chart_table,
        "class",
        "accuracy",
        groupby="model",
        title="RGB vs Flow accuracy per class",
    )
    return table, chart


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

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for (x, flows), y in eval_loader:
                x = x.to(device, non_blocking=True)
                flows = flows.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                B, K = x.shape[:2]
                x = x.view(B * K, *x.shape[2:])

                B_flow, K_flow = flows.shape[:2]
                flows = flows.view(B_flow * K_flow, *flows.shape[2:])

                logits = model.forward(x)
                logits = logits.view(B, K, -1)

                logits_flow = temporal_model.forward(flows)
                logits_flow = logits_flow.view(B, K, -1)

                consensus = logits.mean(dim=1)
                loss = F.cross_entropy(consensus, y)

                consensus_flow = logits_flow.mean(dim=1)
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

        class_table, class_chart = class_comparison_table(
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
            "eval/class_comparison_chart": class_chart,
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

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model.forward(x)
                logits = logits.view(B, K, -1)
                loss = F.cross_entropy(logits.mean(dim=1), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            optimizer_flow.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits_flow = temporal_model.forward(flows)
                logits_flow = logits_flow.view(B, K, -1)
                loss_flow = F.cross_entropy(logits_flow.mean(dim=1), y)
            scaler.scale(loss_flow).backward()
            scaler.step(optimizer_flow)
            scaler.update()

            print(loss_flow.item())

            epoch_loss += loss.item()
            epoch_loss_flow += loss_flow.item()
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

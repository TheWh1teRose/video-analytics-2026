from dataset import UCF101
from model import resnet, resnet_flow
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

NUM_EPOCHS = 10

train_dataset = UCF101("data/data/train.txt", "data/data/mini_UCF", "data/data/mini_UCF_flow")

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
)

eval_dataset = UCF101(
    "data/data/validation.txt",
    "data/data/mini_UCF",
    "data/data/mini_UCF_flow",
    validation=True,
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
)

model = resnet(len(train_dataset.classnames))
temporal_model = resnet_flow(len(train_dataset.classnames))

optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)

optimizer_flow = optim.SGD(temporal_model.parameters(), lr=5e-3, momentum=0.9)
scheduler_flow = optim.lr_scheduler.MultiStepLR(
    optimizer_flow,
    milestones=[12000, 18000],
    gamma=0.1,
)

wandb.init(
    project="TSN",
    config={
        "num_epochs": NUM_EPOCHS,
        "batch_size": 8,
        "lr": 1e-3,
        "lr_flow": 5e-3,
        "num_classes": len(train_dataset.classnames),
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

    with torch.no_grad():
        for (x, flows), y in eval_loader:
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

            total_loss += loss.item()
            total_loss_flow += loss_flow.item()
            correct += (consensus.argmax(dim=1) == y).sum().item()
            correct_flow += (consensus_flow.argmax(dim=1) == y).sum().item()
            total += y.size(0)
            num_batches += 1

    return {
        "eval/loss": total_loss / num_batches,
        "eval/loss_flow": total_loss_flow / num_batches,
        "eval/accuracy": correct / total,
        "eval/accuracy_flow": correct_flow / total,
    }


for epoch in range(NUM_EPOCHS):
    model.train()
    temporal_model.train()
    epoch_loss = 0.0
    epoch_loss_flow = 0.0
    num_batches = 0

    for (x, flows), y in train_loader:
        B, K = x.shape[:2]
        x = x.view(B * K, *x.shape[2:])

        B_flow, K_flow = flows.shape[:2]
        flows = flows.view(B_flow * K_flow, *flows.shape[2:])

        optimizer.zero_grad()
        optimizer_flow.zero_grad()

        logits = model.forward(x)
        logits = logits.view(B, K, -1)

        logits_flow = temporal_model.forward(flows)
        logits_flow = logits_flow.view(B, K, -1)

        consensus = logits.mean(dim=1)
        loss = F.cross_entropy(consensus, y)

        consensus_flow = logits_flow.mean(dim=1)
        loss_flow = F.cross_entropy(consensus_flow, y)

        loss.backward()
        optimizer.step()

        loss_flow.backward()
        optimizer_flow.step()

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

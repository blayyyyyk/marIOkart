from torch.optim.optimizer import Optimizer
from ..models.model_impl import Model
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from typing import Optional


def train_loop(
    model: Model,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    num_epochs: int,
    optimizer: Optimizer,
    device: torch.device,
    report_interval: int = 1,
):
    train_losses, test_losses = [], []
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    for epoch in range(num_epochs):
        # training run
        model.train()
        total_train_loss = 0

        for data in train_dataloader:
            data = {x: y.to(device) for x, y in data.items()}
            targets = data["keymask"].squeeze(2)[..., -1].to(torch.long)
            logits, loss = model(data, targets=targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # eval run (for metrics)
        model.eval()
        total_test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in train_dataloader:
                data = {x: y.to(device) for x, y in data.items()}
                targets = data["keymask"].squeeze(2)[..., -1].to(torch.long)
                logits, loss = model(data, targets=targets)
                
                total_test_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        # report training statistics
        if epoch % report_interval == 0:
            avg_train = total_train_loss / len(train_dataloader)
            avg_test = total_test_loss / len(test_dataloader)
            accuracy = 100 * correct / total

            train_losses.append(avg_train)
            test_losses.append(avg_test)
            print(
                f"Epoch {epoch+1} | Train Loss: {avg_train:.4f} | Test Loss: {avg_test:.4f} | Acc: {accuracy:.2f}%"
            )

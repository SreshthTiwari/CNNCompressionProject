"""train baseline cnn and save it to results/baseline.pt"""
import argparse
import torch
import torch.nn.functional as F
from .data import get_loaders
from .model import SmallCNN


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def accuracy(model, loader, device="cpu"):
    model.eval().to(device)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def train(epochs=10, lr=1e-3, seed=0, out="results/baseline.pt"):
    torch.manual_seed(seed)
    device = get_device()
    train_loader, test_loader = get_loaders()
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        print(f"epoch {ep + 1}/{epochs}  test acc {accuracy(model, test_loader, device):.4f}")
    torch.save(model.cpu().state_dict(), out)
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    train(a.epochs, seed=a.seed)

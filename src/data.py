"""fashion mnist loading and sample visualization"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
MEAN, STD = 0.2860, 0.3530


def get_loaders(batch_size=128, root="data"):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    train = datasets.FashionMNIST(root, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=tf)
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test, batch_size=1000, shuffle=False))


def plot_samples(path="results/samples.png", root="data", per_class=5):
    """save grid with `per_class` example images from each category"""
    import matplotlib.pyplot as plt
    ds = datasets.FashionMNIST(root, train=True, download=True, transform=transforms.ToTensor())
    fig, axes = plt.subplots(10, per_class, figsize=(per_class * 1.2, 10 * 1.2))
    for c in range(10):
        idx = (ds.targets == c).nonzero().flatten()[:per_class]
        for j, i in enumerate(idx):
            ax = axes[c, j]
            ax.imshow(ds[i][0].squeeze(), cmap="gray")
            ax.axis("off")
            if j == 0:
                ax.set_title(CLASSES[c], fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

"""export seed 0 weights and sample test images as json for docs/ web demo"""
import argparse
import base64
import json
import numpy as np
import torch
from torchvision import datasets
from .data import CLASSES, MEAN, STD
from .model import SmallCNN, weight_layers

LAYER_NAMES = ["conv1", "conv2", "fc1", "fc2"]


def b64(arr, dtype):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode()


def export_weights(ckpt, out):
    model = SmallCNN()
    model.load_state_dict(torch.load(ckpt))
    layers = []
    for name, l in zip(LAYER_NAMES, weight_layers(model)):
        layers.append({
            "name": name,
            "shape": list(l.weight.shape),
            "w": b64(l.weight.detach().numpy(), "<f4"),
            "b": b64(l.bias.detach().numpy(), "<f4"),
        })
    json.dump({"layers": layers, "classes": CLASSES, "mean": MEAN, "std": STD}, open(out, "w"))


def export_images(root, per_class, out):
    ds = datasets.FashionMNIST(root, train=False, download=True)
    pixels, labels = [], []
    for c in range(10):
        idx = (ds.targets == c).nonzero().flatten()[:per_class]
        for i in idx:
            pixels.append(ds.data[i].numpy())
            labels.append(c)
    json.dump({"n": len(labels), "labels": labels, "pixels": b64(np.stack(pixels), "u1")}, open(out, "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/baseline_seed0.pt")
    ap.add_argument("--per_class", type=int, default=50)
    a = ap.parse_args()
    export_weights(a.ckpt, "docs/data/weights.json")
    export_images("data", a.per_class, "docs/data/images.json")

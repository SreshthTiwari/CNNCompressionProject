# Neural Network Compression Through Pruning and Quantization

Sreshth Tiwari

## 1. Objective

I'm interested in making machine learning models more efficient. Biological neural networks work with sparse connections, so I wanted to see whether an artificial network can keep its accuracy with fewer connections and lower numerical precision.

## 2. Setup

- Data: Fashion MNIST, 60,000 training and 10,000 test images (28×28 grayscale, 10 clothing classes). Pixels are normalized using the training set mean and std.
- Model: a small CNN with ~200k weights and 91.1% test accuracy. I used a CNN because it learns local visual patterns like edges and shapes, which helps with images.
- Pruning: zero out the smallest weights across the whole network, applied after training with no retraining.
- Quantization: round weights to 8 bit integers. This is simulated: they're converted back to float to run, so it shows the effect on accuracy but not any speed gain.
- Repeats: 5 separate training runs and results are mean ± std.

![samples](results/samples.png)

Samples from each class.

Storage is reported in several ways, because zeros don't shrink a dense file:

| Format | Meaning |
|---|---|
| dense fp32 / int8 | every weight stored, zero or not |
| sparse (ideal) | nonzeros × (value bytes + 4 byte index) |
| zlib | the dense bytes compressed with zlib, an actual file size, with nothing lost |

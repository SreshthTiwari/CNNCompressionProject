# Neural Network Compression Through Pruning and Quantization

Fashion-MNIST + small PyTorch CNN. Compares global magnitude pruning, int8 weight
quantization, and both combined; reports accuracy, nonzero fraction, storage size, and
inference time.

## Setup
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run (from this directory)
```
python -m src.train                # trains baseline (results/baseline.pt)
python -c "from src.data import plot_samples; plot_samples()"   # results/samples.png
python -m src.run_experiments      # results/results.json
python -m src.plots                # results/*.png
python -m src.layer_sparsity       # results/layer_sparsity.{json,png}
```

## Layout
- `src/data.py` loading, normalization, sample grid
- `src/model.py` SmallCNN
- `src/train.py` baseline training + accuracy
- `src/compress.py` pruning, int8 quantization
- `src/metrics.py` sparsity, size (dense / ideal-sparse / zlib), latency
- `src/run_experiments.py`, `src/plots.py`
- `src/layer_sparsity.py` per-layer nonzero fraction under global pruning
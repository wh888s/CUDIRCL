# CUDIRCL

Paper:

**Enhancing diagnostic accuracy in multi-client medical imaging via proxy sharing in unified deep models**

## Overview

This repository provides the PyTorch implementation of CUDIRCL for multi-client CT reconstruction.

## Repository Structure

```text
Datasets/      Data loading and normalization
Fed_model/     Mutual-learning initialization and training utilities
Model/         Reconstruction, image-domain, proxy, loss, and backprojection modules
Solver/        Training, testing, and evaluation code
Utils/         Runtime configuration
main.py        Entry point
```

## Environment

The code requires Python, PyTorch, CUDA-compatible ASTRA Toolbox, and common scientific Python packages.

```bash
pip install -r requirements.txt
```

ASTRA Toolbox is sensitive to CUDA and system configuration. If pip installation does not match your CUDA environment, install it with conda:

```bash
conda install -c astra-toolbox astra-toolbox
```

## Data Format

Set `root_path` in `Utils/initParameter.py`. The expected structure is:

```text
root_path/
  local/
    chest_2e5/
      train/*.mat
      test/*.mat
```

Each `.mat` file should contain:

```text
ndct         Clean CT image, used by supervised clients
fbp          Low-quality reconstruction or low-dose image
sinogram     Sinogram input for reconstruction clients
feature_vec  Client/task feature vector
```

For `unsuper_loss`, the loader uses `fbp` as the label and does not require clean labels for training loss.

## Configuration

Edit `Utils/initParameter.py` before running:

```python
self.root_path = '/path/to/data/'
self.target_path = '/path/to/save/results/'
self.is_train = True
```

The default mixed-task setup is:

```text
client0, client2, client4: reconstruction
client1, client3, client5: denoising
```

Training starts with `client0-client3`; `client4-client5` are added from `join_epoch`.

## Training

```bash
python main.py
```

Training saves checkpoints, optimizer states, losses, validation records, and `.mat` outputs under `target_path`.

## Testing

Set `is_train = False` in `Utils/initParameter.py`, ensure trained checkpoints exist in `target_path/Model_save/`, then run:

```bash
python main.py
```

Test outputs and evaluation metrics are saved under `target_path/<net_name>_result/`.

## License

This project is released under the MIT License. See `LICENSE.txt` for details.

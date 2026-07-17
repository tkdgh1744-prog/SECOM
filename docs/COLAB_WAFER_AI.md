# Colab Wafer AI Runner

The repository code is the single implementation for local CPU, Colab GPU, and
ONNX Runtime CPU execution. Colab only supplies the runtime and dataset path.

## 1. Open the project

Use the collaboration branch until it is merged to `main`:

```python
!git clone -b codex/collaboration-baseline https://github.com/tkdgh1744-prog/SECOM.git
%cd SECOM
```

## 2. Install dependencies

```python
!python -m pip install -r requirements.txt
!python -m pip install -r requirements-ai.txt
```

PyTorch is commonly preinstalled in Colab. Pip will keep a compatible installed
version when it already satisfies the requirement.

## 3. Run the synthetic smoke test

```python
!python scripts/analyze_wafer_maps.py \
  --demo \
  --train-cnn --cnn-epochs 1 \
  --autoencoder --autoencoder-epochs 1 \
  --ai-backend pytorch \
  --device auto \
  --export-onnx \
  --output-dir outputs/wafer_maps_pytorch_demo
```

`--device auto` selects CUDA when Colab provides a GPU and falls back to CPU.

## 4. Connect WM-811K without storing it in Git

Mount Google Drive or use another runtime-visible dataset location:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then pass the dataset path to the same CLI:

```python
!python scripts/analyze_wafer_maps.py \
  --input-path /content/drive/MyDrive/datasets/LSWMD.pkl \
  --input-format wm811k \
  --wafer-map-col waferMap \
  --label-col failureType \
  --group-col lotName \
  --similarity-max-records 0 \
  --train-cnn --cnn-epochs 5 \
  --autoencoder --autoencoder-epochs 5 \
  --ai-backend pytorch \
  --device auto \
  --export-onnx \
  --output-dir outputs/wafer_maps
```

Use `--max-records` for an initial subset check. Omit it for the full run. The
grouped evaluation split keeps each `lotName` entirely in train or test.

## 5. Inspect outputs

```python
import pandas as pd
from IPython.display import display, Image

display(pd.read_csv("outputs/wafer_maps/cnn_metrics.csv"))
display(pd.read_csv("outputs/wafer_maps/cnn_grouped_split.csv").head())
display(Image("outputs/wafer_maps/images/pattern_summary.png"))
```

Important model artifacts:

- `cnn_pattern_classifier.pt`: trusted PyTorch state-dict bundle
- `cnn_pattern_classifier.onnx`: portable CPU inference model
- `wafer_autoencoder.pt`: PyTorch autoencoder bundle
- `wafer_autoencoder.onnx`: portable autoencoder inference model
- `cnn_grouped_split.csv`: auditable lot-level split assignment
- `cnn_metrics.csv`: evaluation result and model parameter count

Raw datasets and generated model files remain excluded from Git.

## 6. Profile the deployable CPU model

Run this step on the target CPU runtime, even when training used a Colab GPU:

```python
!python scripts/profile_wafer_models.py \
  --pytorch-model outputs/wafer_maps/cnn_pattern_classifier.pt \
  --onnx-model outputs/wafer_maps/cnn_pattern_classifier.onnx \
  --quantize-int8 \
  --batch-size 1 \
  --warmup-runs 10 \
  --measured-runs 50 \
  --intra-op-threads 1 \
  --output-dir outputs/profiling/wafer_cnn
```

The JSON report records the runtime and hardware context. NPU results must stay
marked unavailable until a concrete device and runtime can execute the same
measurement protocol.

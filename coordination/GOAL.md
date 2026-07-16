# Project Goal

## Mission

Build an educational semiconductor AI system that combines three analysis tracks in a
single program and then evaluates them from an AI-semiconductor hardware perspective.

## Analysis tracks

1. SECOM AI: process sensor values to pass/fail prediction. CPU execution is the initial target.
2. Wafer-map AI: wafer images to defect-pattern classification (PyTorch, per D-009). GPU may be used for training convenience only; the optimization target is CPU / ONNX Runtime (D-010).
3. Equipment AI: time-series sensor values to anomaly-time detection. Small models target CPU;
   larger deep-learning models may target GPU.
4. Integrated program: present the three results in one consistent interface.

## Development stages

1. Secure and validate SECOM, WM-811K, and equipment time-series data.
2. Build reproducible classification and anomaly-detection baselines.
3. Integrate predictions, confidence, quality indicators, and monitoring results.
4. Profile operation count, model size, latency, memory use, and power-related proxies.
5. Profile primarily on CPU / ONNX Runtime (the optimization target, D-010); compare GPU or NPU only where concrete hardware and a runtime are available.
6. Apply quantization, pruning or distillation where useful, operation reduction, and
   memory-access optimization.

## First milestone

The first milestone is complete when CI passes, each analysis track has a reproducible
CPU baseline or clearly marked demo, and one integrated report or screen can load all
available outputs without implying that unrelated records are linked.

## Current data boundary

- SECOM download and processing code exists.
- Demo wafer-map outputs exist.
- The real WM-811K dataset is not currently present in the local repository.
- Equipment time-series inputs must be obtained or generated under an explicit demo label.

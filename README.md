# SECOM Semiconductor Process Data Analysis

UCI SECOM semiconductor process sensor data瑜??ъ슜???쒗뭹???뺤긽/遺덈웾???덉륫?섍퀬, ?댄썑 ?⑥씠??寃???곗씠?곗? ?ㅻ퉬 ?대깽???곗씠?곕줈 ?뺤옣?????덇쾶 留뚮뱺 遺꾩꽍 ?명듃遺곸엯?덈떎.

## Current Artifact

- `SECOM.ipynb`: main Google Colab analysis notebook
- `requirements.txt`: package list for local execution
- `requirements-ai.txt`: optional PyTorch, ONNX, and ONNX Runtime stack
- `Makefile`: standard project commands
- `src/secom_data.py`: reusable data loading utilities
- `src/data_contracts.py`: schema and value contract checks for input/output tables
- `src/quality_reports.py`: standard data quality report utilities
- `src/model_registry.py`: model bundle save/load and prediction utilities
- `src/monitoring.py`: process-quality prediction monitoring utilities
- `src/reporting.py`: Markdown summary report utilities
- `src/secom_modeling.py`: reusable preprocessing, prediction, and evaluation utilities
- `src/secom_training.py`: model ranking and threshold tuning helpers
- `src/wafer_features.py`: wafer defect spatial feature utilities
- `src/wafer_map_analysis.py`: wafer map spatial pattern analysis, similarity, clustering, visualization, and optional AI utilities
- `src/wafer_torch.py`: grouped PyTorch training, model bundles, and ONNX export
- `src/wafer_ai_outputs.py`: PyTorch and legacy TensorFlow artifact orchestration
- `src/model_profiling.py`: repeatable PyTorch/ONNX CPU latency, memory, and operation profiling
- `src/equipment_features.py`: equipment event feature utilities
- `src/equipment_anomaly.py`: time-aware robust equipment anomaly detector
- `src/integrated_dashboard.py`: standalone integrated result dashboard builder
- `src/feature_store.py`: sensor, wafer, and equipment feature assembly utilities
- `scripts/run_quality_report.py`: CLI for writing SECOM quality report CSV files
- `scripts/train_secom_model.py`: CLI for training, selecting, tuning, and saving SECOM models
- `scripts/build_auxiliary_features.py`: CLI for building wafer/equipment feature CSV files
- `scripts/analyze_wafer_maps.py`: CLI for WM-811K/array/coordinate wafer map analysis
- `scripts/profile_wafer_models.py`: CLI for FP32/INT8 wafer-model profiling
- `scripts/analyze_equipment_anomalies.py`: CLI for equipment sensor anomaly detection
- `scripts/generate_integrated_dashboard.py`: CLI for generating the integrated HTML dashboard
- `docs/COLAB_WAFER_AI.md`: Colab runner and external WM-811K path guide
- `docs/AGENT_BRIDGE.md`: bounded hosted Codex-Claude review-loop design
- `scripts/assemble_feature_table.py`: CLI for assembling modeling feature tables
- `scripts/predict_with_model.py`: CLI for batch prediction with saved model bundles
- `scripts/generate_monitoring_report.py`: CLI for process-quality monitoring reports
- `scripts/generate_summary_report.py`: CLI for Markdown summary reports
- `scripts/run_pipeline.py`: CLI for orchestrating quality, feature assembly, prediction, and monitoring steps
- `tests/`: synthetic-data tests for contracts, loading, quality, reporting, modeling, model registry, monitoring, wafer, equipment, feature-store, pipeline, and CLI utilities

## Implemented Scope

- SECOM ?먮낯 ?곗씠???먮룞 ?ㅼ슫濡쒕뱶
- ?쇱꽌 Feature? ?쇰꺼 濡쒕뵫
- ?먮낯 ?쇰꺼 `-1/1`??`0=Pass`, `1=Fail`濡?蹂??
- ?좎쭨/?쒓컙 ?뺣낫 蹂꾨룄 蹂닿?
- 寃곗륫媛? ?대옒??遺덇퇏?? ?곸닔/?遺꾩궛 Feature 遺꾩꽍
- 二쇱슂 Feature 遺꾪룷, Pass/Fail 鍮꾧탳, ?곴?愿怨? PCA ?쒓컖??
- Train/Test 遺꾨━
- ?곗씠???꾩닔 諛⑹? Pipeline
- Dummy Classifier, Logistic Regression, Random Forest 鍮꾧탳
- Class Weight, Random Oversampling, SMOTE
- Stratified Cross Validation
- Threshold 遺꾩꽍
- ?쇰룞?됰젹, ROC Curve, Precision-Recall Curve
- Feature Importance, Permutation Importance
- ?좏깮??MLP ?좉꼍留?鍮꾧탳
- 理쒖쥌 Pipeline ??κ낵 ???쇱꽌 ?곗씠???덉륫 ?⑥닔
- ?⑥씠??寃???ㅻ퉬 ?대깽???곗씠???좏깮??濡쒕뜑

## Colab Usage

1. `SECOM.ipynb`瑜?Google Colab?먯꽌 ?쎈땲??
2. 硫붾돱?먯꽌 `Runtime > Run all`???ㅽ뻾?⑸땲??
3. `imbalanced-learn` ?먮뒗 `tensorflow`媛 ?녿뒗 ?고??꾩씠硫??대떦 ?좏깮???뱀뀡? ?덈궡 硫붿떆吏瑜?異쒕젰?⑸땲??
4. ?ㅽ뻾 寃곌낵? 紐⑤뜽 ?뚯씪? ?명듃遺?湲곗? `outputs/` ?꾨옒???앹꽦?⑸땲??

## Local Usage

```bash
pip install -r requirements.txt
jupyter notebook SECOM.ipynb
```

## Standard Commands

```bash
make check              # validate notebook structure and run tests
make train-secom        # train, evaluate, tune threshold, and save a SECOM model
make quality-report     # generate SECOM quality CSV reports
make auxiliary-features # build wafer/equipment feature CSV files
make wafer-map-analysis # analyze WM-811K-style wafer map spatial patterns
make wafer-map-demo     # generate deterministic synthetic wafer-map outputs
make wafer-ai-demo      # train PyTorch demo models and export ONNX
make profile-wafer-ai   # compare PyTorch, ONNX FP32, and ONNX INT8 on CPU
make equipment-anomaly-demo # generate deterministic equipment anomaly outputs
make dashboard          # generate the integrated dashboard from default output paths
make dashboard-demo     # generate both synthetic tracks and the integrated dashboard
make assemble-features  # assemble one modeling feature table
make predict            # run batch prediction with a saved model bundle
make monitor            # generate monitoring CSV reports
make summary            # generate Markdown summary report
make pipeline           # run the full CLI pipeline
```

## Model Training CLI

Train candidate models, choose the best validation model, tune the decision threshold, and save the final model bundle plus CSV reports:

```bash
python scripts/train_secom_model.py --download-secom
```

Main outputs:

- `outputs/models/secom_final_pipeline.joblib`
- `outputs/reports/model_metrics.csv`
- `outputs/reports/threshold_metrics.csv`
- `outputs/predictions/test_predictions.csv`

## Quality Report CLI

Generate standard SECOM quality report CSV files with local raw files:

```bash
python scripts/run_quality_report.py --raw-data-dir data/raw --output-dir outputs/reports
```

Download the UCI SECOM files first, then write reports:

```bash
python scripts/run_quality_report.py --download
```

## Auxiliary Feature CLI

Build wafer and equipment feature CSV files from raw auxiliary tables:

```bash
python scripts/build_auxiliary_features.py --wafer-input data/raw/wafer_inspection.csv --equipment-input data/raw/equipment_events.csv --wafer-output data/raw/wafer_features.csv --equipment-output data/raw/equipment_features.csv --add-wafer-pattern-label
```


## Wafer Map Analysis CLI

WM-811K 스타일 pickle(`waferMap` 컬럼), `.npy/.npz` 배열, 또는 `wafer_id,x,y,value/is_defect` 좌표 CSV를 입력으로 받아 공간 불량 패턴 분석 프로그램을 실행합니다.

```bash
python scripts/analyze_wafer_maps.py --demo --output-dir outputs/wafer_maps_demo
python scripts/analyze_wafer_maps.py --input-path data/raw/wm811k.pkl --input-format wm811k --output-dir outputs/wafer_maps
python scripts/analyze_wafer_maps.py --input-path data/raw/wafer_die_map.csv --input-format coordinate --value-col die_status --coordinate-defect-value FAIL --output-dir outputs/wafer_maps
```

Install the optional AI stack, then use the same command on local CPU or Colab GPU:

```bash
python -m pip install -r requirements-ai.txt
python scripts/analyze_wafer_maps.py --demo --train-cnn --cnn-epochs 1 --autoencoder --autoencoder-epochs 1 --ai-backend pytorch --device auto --export-onnx --output-dir outputs/wafer_maps_pytorch_demo
```

See `docs/COLAB_WAFER_AI.md` for Google Drive dataset mounting and a full WM-811K command.

주요 산출물:

- `outputs/wafer_maps/wafer_map_features.csv`: 불량 Die 개수/비율, 중심부/외곽부 비율, 반경별 분포, 군집 수, 대칭성, 방향 편향, 휴리스틱 패턴
- `outputs/wafer_maps/pattern_summary.csv`: Center, Edge-Ring, Edge-Local, Scratch, Donut, Local, Random, Near-Full 요약
- `outputs/wafer_maps/similar_wafer_pairs.csv`: 유사 웨이퍼 검색 결과
- `outputs/wafer_maps/cluster_summary.csv`: 알려지지 않은 패턴 군집 요약
- `outputs/wafer_maps/images/`: 웨이퍼 맵 및 패턴 요약 PNG
- `outputs/wafer_maps/wafer_map_report.md`: 분석 리포트

PyTorch is the default AI backend. `--train-cnn` writes a `.pt` state-dict bundle,
grouped split audit CSV, metrics, and optional ONNX model. `--autoencoder` writes
reconstruction scores plus `.pt` and ONNX models. WM-811K uses `lotName` as the
default leakage group. The temporary legacy path remains available with
`--ai-backend tensorflow`.

## Wafer Model Profiling CLI

Profile the saved PyTorch bundle and ONNX model with repeated CPU measurements,
then create and measure a dynamically weight-quantized INT8 ONNX model:

```bash
python scripts/profile_wafer_models.py --pytorch-model outputs/wafer_maps_pytorch_demo/cnn_pattern_classifier.pt --onnx-model outputs/wafer_maps_pytorch_demo/cnn_pattern_classifier.onnx --quantize-int8 --output-dir outputs/profiling/wafer_cnn
```

The profiler writes `model_profiles.csv`, `operation_profiles.csv`, and
`model_profiles.json`. Results include model and parameter bytes, input/output
memory proxies, process RSS observations, operator counts, repeated latency
percentiles, and throughput. CPU thread count, batch size, warmup count, and
measurement count are explicit CLI settings so later hardware comparisons use
the same protocol.

## Equipment Anomaly CLI

Run the deterministic CPU demo or analyze a real time-ordered sensor CSV:

```bash
python scripts/analyze_equipment_anomalies.py --demo --output-dir outputs/equipment_anomalies_demo
python scripts/analyze_equipment_anomalies.py --input-path data/raw/equipment_sensors.csv --sensor-columns temperature,vibration,pressure --label-col failure_label --integration-mode real --output-dir outputs/equipment_anomalies
```

Main outputs:

- `equipment_anomaly_scores.csv`: time-ordered anomaly scores and split labels
- `equipment_anomaly_summary.csv`: equipment and split summaries
- `equipment_anomaly_metrics.csv`: evaluation metrics when labels are available
- `equipment_anomaly_model.pkl`: reusable robust z-score detector bundle
- `equipment_anomaly_metadata.json`: detector settings, threshold, and provenance mode

## Integrated Dashboard CLI

Generate a standalone HTML view from any available SECOM, wafer, and equipment outputs:

```bash
python scripts/generate_integrated_dashboard.py
python scripts/generate_integrated_dashboard.py --wafer-dir outputs/wafer_maps_demo --wafer-mode synthetic --equipment-dir outputs/equipment_anomalies_demo --equipment-mode synthetic
```

The output is written to `outputs/integrated_dashboard/index.html`. Missing tracks remain visible as unavailable, and the dashboard never joins unrelated records by row order.

## Feature Assembly CLI

Assemble sensor, wafer, and equipment feature CSV files into one modeling table:

```bash
python scripts/assemble_feature_table.py --sensor-path data/raw/sensor_features.csv --wafer-path data/raw/wafer_features.csv --equipment-path data/raw/equipment_features.csv --output-path outputs/features/modeling_table.csv
```

## Prediction CLI

Run batch predictions with a saved model bundle and a feature CSV file:

```bash
python scripts/predict_with_model.py --model-path outputs/models/secom_final_pipeline.joblib --features-path outputs/features/modeling_table.csv --output-path outputs/predictions/predictions.csv --id-columns sample_id,wafer_id
```

## Monitoring CLI

Generate process-quality monitoring reports from prediction outputs:

```bash
python scripts/generate_monitoring_report.py --predictions-path outputs/predictions/predictions.csv --group-columns wafer_id,equipment_id --output-dir outputs/reports/monitoring
```

## End-to-End Pipeline CLI

Run quality reporting, feature assembly, prediction, and monitoring in one command:

```bash
python scripts/run_pipeline.py --sensor-path data/raw/sensor_features.csv --wafer-path data/raw/wafer_features.csv --equipment-path data/raw/equipment_features.csv --model-path outputs/models/secom_final_pipeline.joblib --id-columns sample_id,wafer_id --monitoring-group-columns wafer_id,equipment_id
```

## Summary Report CLI

Generate a Markdown summary from available CSV reports:

```bash
python scripts/generate_summary_report.py --reports-dir outputs/reports --monitoring-dir outputs/reports/monitoring --output-path outputs/reports/summary_report.md
```

## Validation

Notebook 援ъ“? 肄붾뱶 ? 臾몃쾿??鍮좊Ⅴ寃??뺤씤?섎젮硫??ㅼ쓬 紐낅졊???ㅽ뻾?⑸땲??

```bash
python validate_notebook.py
python -m unittest discover -s tests
```

## Data Policy

?먮낯 ?곗씠?곗? ?ㅽ뻾 ?곗텧臾쇱? Git???щ━吏 ?딆뒿?덈떎.

- `data/raw/`
- `outputs/`

## Next Data Extensions

異붽? ?곗씠?곌? ?뺣낫?섎㈃ ?ㅼ쓬 ?꾩튂???ｊ퀬 ?명듃遺곸쓽 ?좏깮??濡쒕뜑 ?뱀뀡???ㅽ뻾?⑸땲??

- `data/raw/wafer_inspection.csv`
- `data/raw/equipment_events.csv`

?⑥씠??寃고븿 ?⑦꽩 遺꾨쪟? ?ㅻ퉬 怨좎옣 ?덉륫? ?ㅼ젣 而щ읆 援ъ“媛 ?뺤씤????蹂꾨룄 Feature Engineering怨??쒓컙 湲곗? 寃利앹쓣 異붽??댁빞 ?⑸땲??



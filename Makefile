PYTHON ?= python

.PHONY: validate test check train-secom quality-report auxiliary-features wafer-map-analysis assemble-features predict monitor summary pipeline

validate:
	$(PYTHON) validate_notebook.py

test:
	$(PYTHON) -m unittest discover -s tests

check: validate test

train-secom:
	$(PYTHON) scripts/train_secom_model.py --download-secom --raw-data-dir data/raw --model-output outputs/models/secom_final_pipeline.joblib --metrics-output outputs/reports/model_metrics.csv --threshold-output outputs/reports/threshold_metrics.csv --predictions-output outputs/predictions/test_predictions.csv

quality-report:
	$(PYTHON) scripts/run_quality_report.py --raw-data-dir data/raw --output-dir outputs/reports/quality

auxiliary-features:
	$(PYTHON) scripts/build_auxiliary_features.py --wafer-input data/raw/wafer_inspection.csv --equipment-input data/raw/equipment_events.csv --wafer-output data/raw/wafer_features.csv --equipment-output data/raw/equipment_features.csv --add-wafer-pattern-label


wafer-map-analysis:
	$(PYTHON) scripts/analyze_wafer_maps.py --input-path data/raw/wm811k.pkl --input-format wm811k --output-dir outputs/wafer_maps

assemble-features:
	$(PYTHON) scripts/assemble_feature_table.py --sensor-path data/raw/sensor_features.csv --wafer-path data/raw/wafer_features.csv --equipment-path data/raw/equipment_features.csv --output-path outputs/features/modeling_table.csv

predict:
	$(PYTHON) scripts/predict_with_model.py --model-path outputs/models/secom_final_pipeline.joblib --features-path outputs/features/modeling_table.csv --output-path outputs/predictions/predictions.csv --id-columns sample_id,wafer_id

monitor:
	$(PYTHON) scripts/generate_monitoring_report.py --predictions-path outputs/predictions/predictions.csv --group-columns wafer_id,equipment_id --output-dir outputs/reports/monitoring

summary:
	$(PYTHON) scripts/generate_summary_report.py --reports-dir outputs/reports --monitoring-dir outputs/reports/monitoring --output-path outputs/reports/summary_report.md

pipeline:
	$(PYTHON) scripts/run_pipeline.py --sensor-path data/raw/sensor_features.csv --wafer-path data/raw/wafer_features.csv --equipment-path data/raw/equipment_features.csv --model-path outputs/models/secom_final_pipeline.joblib --id-columns sample_id,wafer_id --monitoring-group-columns wafer_id,equipment_id



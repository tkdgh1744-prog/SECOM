"""Validate the SECOM analysis notebook structure and Python syntax."""

from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOK_PATH = Path("SECOM.ipynb")

REQUIRED_FILES = [
    Path("README.md"),
    Path("requirements.txt"),
    Path("Makefile"),
    Path("src/secom_data.py"),
    Path("src/data_contracts.py"),
    Path("src/secom_modeling.py"),
    Path("src/wafer_features.py"),
    Path("src/equipment_features.py"),
    Path("src/feature_store.py"),
    Path("src/quality_reports.py"),
    Path("src/model_registry.py"),
    Path("src/monitoring.py"),
    Path("src/reporting.py"),
    Path("src/secom_training.py"),
    Path("scripts/run_quality_report.py"),
    Path("scripts/train_secom_model.py"),
    Path("scripts/assemble_feature_table.py"),
    Path("scripts/build_auxiliary_features.py"),
    Path("scripts/predict_with_model.py"),
    Path("scripts/generate_monitoring_report.py"),
    Path("scripts/generate_summary_report.py"),
    Path("scripts/run_pipeline.py"),
    Path("tests/test_secom_data.py"),
    Path("tests/test_data_contracts.py"),
    Path("tests/test_secom_modeling.py"),
    Path("tests/test_secom_training.py"),
    Path("tests/test_wafer_features.py"),
    Path("tests/test_equipment_features.py"),
    Path("tests/test_feature_store.py"),
    Path("tests/test_quality_reports.py"),
    Path("tests/test_model_registry.py"),
    Path("tests/test_monitoring.py"),
    Path("tests/test_reporting.py"),
    Path("tests/test_run_quality_report.py"),
    Path("tests/test_assemble_feature_table.py"),
    Path("tests/test_build_auxiliary_features.py"),
    Path("tests/test_predict_with_model.py"),
    Path("tests/test_generate_monitoring_report.py"),
    Path("tests/test_run_pipeline.py"),
    Path("tests/test_train_secom_model.py"),
]

REQUIRED_SECTIONS = [
    "## 1. 라이브러리 Import",
    "## 2. Seed와 경로 설정",
    "## 3. SECOM 데이터 다운로드",
    "## 4. 데이터와 라벨 불러오기",
    "## 5. 데이터 구조 확인",
    "## 6. 정상·불량 클래스 분포",
    "## 7. 결측값 1차 확인",
    "## 8. 기초 시각화",
    "## 9. 상수·저분산 Feature 분석",
    "## 10. 센서 Feature 분포 시각화",
    "## 11. 정상·불량별 Feature 비교",
    "## 12. Feature 간 상관관계 분석",
    "## 13. PCA 시각화",
    "## 14. Train/Test 분리",
    "## 15. 전처리 Pipeline 구성",
    "## 16. Dummy Classifier",
    "## 17. Logistic Regression",
    "## 18. Random Forest",
    "## 19. 모델 성능 비교",
    "## 20. 혼동행렬",
    "## 21. ROC Curve와 Precision-Recall Curve",
    "## 22. Threshold 분석",
    "## 23. 최종 후보 모델 평가",
    "## 24. Feature Importance",
    "## 25. 새로운 데이터 예측",
    "## 26. 모델 저장",
    "## 27. 결과 요약과 한계",
    "## 28. 웨이퍼 검사·설비 이벤트 데이터 확장 인터페이스",
    "## 29. 다음 확장 방향",
    "## 30. 클래스 불균형 처리 심화",
    "## 31. Stratified Cross Validation",
    "## 32. MLP 신경망 비교",
]

REQUIRED_TOKENS = [
    'sep=r"\\s+"',
    "stratify=y",
    "Pipeline",
    "HighMissingFeatureDropper",
    "DummyClassifier",
    "LogisticRegression",
    "RandomForestClassifier",
    "SMOTE",
    "StratifiedKFold",
    "predict_secom",
    "joblib.dump",
    "wafer_inspection",
    "equipment_events",
]


def load_notebook(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Notebook not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def cell_source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def validate_sections(markdown_text: str) -> list[str]:
    return [section for section in REQUIRED_SECTIONS if section not in markdown_text]


def validate_tokens(code_text: str) -> list[str]:
    return [token for token in REQUIRED_TOKENS if token not in code_text]


def validate_code_syntax(code_cells: list[str]) -> list[str]:
    errors = []
    for idx, source in enumerate(code_cells, start=1):
        try:
            ast.parse(source)
        except SyntaxError as exc:
            errors.append(f"Code cell {idx}: {exc}")
    return errors


def validate_required_files() -> list[str]:
    return [str(path) for path in REQUIRED_FILES if not path.exists()]


def main() -> int:
    notebook = load_notebook(NOTEBOOK_PATH)
    cells = notebook.get("cells", [])
    markdown_cells = [cell_source(cell) for cell in cells if cell.get("cell_type") == "markdown"]
    code_cells = [cell_source(cell) for cell in cells if cell.get("cell_type") == "code"]

    markdown_text = "\n".join(markdown_cells)
    code_text = "\n\n".join(code_cells)

    missing_sections = validate_sections(markdown_text)
    missing_tokens = validate_tokens(code_text)
    syntax_errors = validate_code_syntax(code_cells)
    missing_files = validate_required_files()

    print(f"Notebook: {NOTEBOOK_PATH}")
    print(f"Total cells: {len(cells)}")
    print(f"Markdown cells: {len(markdown_cells)}")
    print(f"Code cells: {len(code_cells)}")

    if missing_sections:
        print("\nMissing sections:")
        for section in missing_sections:
            print(f"- {section}")

    if missing_tokens:
        print("\nMissing required code tokens:")
        for token in missing_tokens:
            print(f"- {token}")

    if syntax_errors:
        print("\nSyntax errors:")
        for error in syntax_errors:
            print(f"- {error}")

    if missing_files:
        print("\nMissing project files:")
        for file_path in missing_files:
            print(f"- {file_path}")

    if missing_sections or missing_tokens or syntax_errors or missing_files:
        print("\nValidation failed.")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

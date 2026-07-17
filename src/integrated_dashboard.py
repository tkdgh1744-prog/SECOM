"""Build a standalone dashboard from SECOM, wafer, and equipment outputs."""

from __future__ import annotations

import base64
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd


VALID_MODES = {"unknown", "synthetic", "demo", "real"}


def _read_csv(path: Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def _read_json(path: Path) -> dict[str, object]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_mode(mode: object) -> str:
    text = str(mode or "unknown").strip().lower()
    return text if text in VALID_MODES else "unknown"


def _format_value(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    if isinstance(value, (float, np.floating)):
        absolute = abs(float(value))
        if absolute >= 1000:
            return f"{float(value):,.0f}"
        if absolute >= 10:
            return f"{float(value):.2f}"
        return f"{float(value):.3f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def _table_html(
    frame: pd.DataFrame | None,
    columns: list[str] | None = None,
    max_rows: int = 10,
) -> str:
    if frame is None or frame.empty:
        return '<div class="empty">No data available.</div>'
    display = frame.copy()
    if columns:
        usable = [column for column in columns if column in display.columns]
        if usable:
            display = display[usable]
    display = display.head(max_rows)
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in display.columns)
    rows = []
    for row in display.itertuples(index=False, name=None):
        cells = "".join(f"<td>{escape(_format_value(value))}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + headers
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _metric_html(label: str, value: object, tone: str = "neutral") -> str:
    return (
        f'<div class="metric metric-{escape(tone)}">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{escape(_format_value(value))}</div>'
        "</div>"
    )


def _mode_badge(mode: str) -> str:
    normalized = _normalize_mode(mode)
    return f'<span class="mode mode-{normalized}">{escape(normalized.upper())}</span>'


def _embedded_png(path: Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:image/png;base64,{encoded}"


def _first_numeric(frame: pd.DataFrame | None, column: str) -> float | None:
    if frame is None or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else None


def _secom_panel(reports_dir: Path, mode: str) -> tuple[str, dict[str, object]]:
    overview = _read_csv(reports_dir / "quality" / "overview.csv")
    class_distribution = _read_csv(reports_dir / "quality" / "class_distribution.csv")
    model_metrics = _read_csv(reports_dir / "model_metrics.csv")
    risk_summary = _read_csv(reports_dir / "monitoring" / "overall_risk_summary.csv")
    available = any(frame is not None for frame in (overview, class_distribution, model_metrics, risk_summary))

    sample_count = None
    if overview is not None and {"metric", "value"}.issubset(overview.columns):
        match = overview.loc[overview["metric"].astype(str).str.lower().eq("n_samples"), "value"]
        if not match.empty:
            sample_count = match.iloc[0]

    fail_count = None
    if class_distribution is not None and {"class", "count"}.issubset(class_distribution.columns):
        failures = class_distribution.loc[
            class_distribution["class"].astype(str).str.lower().isin(["fail", "failure", "1"]),
            "count",
        ]
        if not failures.empty:
            fail_count = failures.iloc[0]

    metrics = "".join(
        [
            _metric_html("Samples", sample_count),
            _metric_html("Failures", fail_count, "danger" if fail_count else "neutral"),
            _metric_html("Model rows", 0 if model_metrics is None else len(model_metrics)),
            _metric_html("Risk summaries", 0 if risk_summary is None else len(risk_summary)),
        ]
    )
    panel = f"""
    <section class="track-head">
      <div><div class="eyebrow">PROCESS SENSOR CLASSIFICATION</div><h2>SECOM AI</h2></div>
      {_mode_badge(mode)}
    </section>
    <div class="metric-grid">{metrics}</div>
    <div class="section-grid">
      <section><h3>Model metrics</h3>{_table_html(model_metrics)}</section>
      <section><h3>Class distribution</h3>{_table_html(class_distribution)}</section>
      <section><h3>Quality overview</h3>{_table_html(overview)}</section>
      <section><h3>Risk summary</h3>{_table_html(risk_summary)}</section>
    </div>
    """
    return panel, {"available": available, "mode": _normalize_mode(mode)}


def _wafer_panel(wafer_dir: Path, mode: str) -> tuple[str, dict[str, object]]:
    patterns = _read_csv(wafer_dir / "pattern_summary.csv")
    features = _read_csv(wafer_dir / "wafer_map_features.csv")
    clusters = _read_csv(wafer_dir / "cluster_summary.csv")
    available = any(frame is not None for frame in (patterns, features, clusters))
    wafer_count = len(features) if features is not None else None
    if wafer_count is None and patterns is not None and "wafer_count" in patterns.columns:
        wafer_count = int(pd.to_numeric(patterns["wafer_count"], errors="coerce").fillna(0).sum())

    top_pattern = None
    if patterns is not None and not patterns.empty and "heuristic_pattern" in patterns.columns:
        ordered = patterns
        if "wafer_count" in patterns.columns:
            ordered = patterns.sort_values("wafer_count", ascending=False)
        top_pattern = ordered.iloc[0]["heuristic_pattern"]
    mean_defect = None
    if features is not None and "defect_ratio" in features.columns:
        mean_defect = pd.to_numeric(features["defect_ratio"], errors="coerce").mean()

    image_src = _embedded_png(wafer_dir / "images" / "pattern_summary.png")
    image_html = (
        f'<img class="wafer-image" src="{image_src}" alt="Wafer pattern distribution">'
        if image_src
        else '<div class="empty visual-empty">No wafer chart available.</div>'
    )
    metrics = "".join(
        [
            _metric_html("Wafer maps", wafer_count),
            _metric_html("Top pattern", top_pattern),
            _metric_html("Mean defect ratio", mean_defect, "danger" if mean_defect else "neutral"),
            _metric_html("Clusters", 0 if clusters is None else len(clusters)),
        ]
    )
    panel = f"""
    <section class="track-head">
      <div><div class="eyebrow">SPATIAL DEFECT ANALYSIS</div><h2>Wafer Map AI</h2></div>
      {_mode_badge(mode)}
    </section>
    <div class="metric-grid">{metrics}</div>
    <div class="wafer-layout">
      <section><h3>Pattern distribution</h3>{image_html}</section>
      <section><h3>Pattern summary</h3>{_table_html(patterns)}</section>
    </div>
    <section><h3>Highest defect ratios</h3>{_table_html(features.sort_values("defect_ratio", ascending=False) if features is not None and "defect_ratio" in features.columns else features, ["wafer_id", "defect_ratio", "defect_die_count", "heuristic_pattern", "source_label"])}</section>
    """
    return panel, {"available": available, "mode": _normalize_mode(mode)}


def _equipment_panel(equipment_dir: Path, fallback_mode: str) -> tuple[str, dict[str, object]]:
    metadata = _read_json(equipment_dir / "equipment_anomaly_metadata.json")
    metrics_frame = _read_csv(equipment_dir / "equipment_anomaly_metrics.csv")
    summary = _read_csv(equipment_dir / "equipment_anomaly_summary.csv")
    scores = _read_csv(equipment_dir / "equipment_anomaly_scores.csv")
    mode = _normalize_mode(metadata.get("integration_mode", fallback_mode))
    available = any(frame is not None for frame in (metrics_frame, summary, scores))

    anomaly_count = None
    evaluation_rows = metadata.get("evaluation_rows")
    if scores is not None and "is_anomaly" in scores.columns:
        anomaly_rows = scores
        if "split" in scores.columns:
            evaluation_mask = scores["split"].astype(str).str.lower().isin(["evaluation", "eval", "test"])
            if evaluation_mask.any():
                anomaly_rows = scores.loc[evaluation_mask]
        anomaly_count = int(
            anomaly_rows["is_anomaly"].astype(str).str.lower().isin(["true", "1"]).sum()
        )
    f1 = _first_numeric(metrics_frame, "f1")
    threshold = metadata.get("threshold")
    metrics = "".join(
        [
            _metric_html("Evaluation rows", evaluation_rows),
            _metric_html("Evaluation anomalies", anomaly_count, "danger" if anomaly_count else "neutral"),
            _metric_html("Evaluation F1", f1),
            _metric_html("Threshold", threshold),
        ]
    )

    chart_rows: list[dict[str, object]] = []
    if scores is not None and {"timestamp", "anomaly_score"}.issubset(scores.columns):
        chart_source = scores.sort_values("timestamp").tail(400)
        for _, row in chart_source.iterrows():
            chart_rows.append(
                {
                    "timestamp": str(row["timestamp"]),
                    "score": float(pd.to_numeric(row["anomaly_score"], errors="coerce")),
                    "anomaly": str(row.get("is_anomaly", "")).lower() in {"true", "1"},
                    "equipment": str(row.get("equipment_id", "")),
                }
            )
    panel = f"""
    <section class="track-head">
      <div><div class="eyebrow">TIME-ORDERED SENSOR MONITORING</div><h2>Equipment AI</h2></div>
      {_mode_badge(mode)}
    </section>
    <div class="metric-grid">{metrics}</div>
    <section>
      <h3>Anomaly score timeline</h3>
      <div class="chart-shell"><canvas id="equipment-chart" height="280"></canvas></div>
    </section>
    <div class="section-grid">
      <section><h3>Equipment summary</h3>{_table_html(summary)}</section>
      <section><h3>Evaluation metrics</h3>{_table_html(metrics_frame)}</section>
    </div>
    """
    return panel, {
        "available": available,
        "mode": mode,
        "chart_rows": chart_rows,
    }


def build_integrated_dashboard(
    reports_dir: Path = Path("outputs/reports"),
    wafer_dir: Path = Path("outputs/wafer_maps"),
    equipment_dir: Path = Path("outputs/equipment_anomalies"),
    output_path: Path = Path("outputs/integrated_dashboard/index.html"),
    secom_mode: str = "unknown",
    wafer_mode: str = "unknown",
    equipment_mode: str = "unknown",
) -> str:
    """Build and write a standalone four-tab manufacturing AI dashboard."""
    reports_dir = Path(reports_dir)
    wafer_dir = Path(wafer_dir)
    equipment_dir = Path(equipment_dir)
    output_path = Path(output_path)
    secom_panel, secom_state = _secom_panel(reports_dir, secom_mode)
    wafer_panel, wafer_state = _wafer_panel(wafer_dir, wafer_mode)
    equipment_panel, equipment_state = _equipment_panel(equipment_dir, equipment_mode)

    states = [
        ("SECOM AI", secom_state),
        ("Wafer Map AI", wafer_state),
        ("Equipment AI", equipment_state),
    ]
    availability_rows = pd.DataFrame(
        [
            {
                "track": name,
                "status": "available" if state["available"] else "not available",
                "integration_mode": state["mode"],
            }
            for name, state in states
        ]
    )
    available_count = sum(bool(state["available"]) for _, state in states)
    overview_metrics = "".join(
        [
            _metric_html("Available tracks", f"{available_count} / 3"),
            _metric_html("SECOM", "ready" if secom_state["available"] else "missing"),
            _metric_html("Wafer", "ready" if wafer_state["available"] else "missing"),
            _metric_html("Equipment", "ready" if equipment_state["available"] else "missing"),
        ]
    )
    chart_payload = json.dumps(equipment_state["chart_rows"], ensure_ascii=True).replace("<", "\\u003c")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Semiconductor AI Operations</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f6f8;
      --surface: #ffffff;
      --line: #d9dee5;
      --text: #18212b;
      --muted: #647181;
      --blue: #1769aa;
      --green: #247a52;
      --red: #bd3b3b;
      --amber: #9a6700;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font: 14px/1.5 Arial, sans-serif; letter-spacing: 0; }}
    header {{ background: #15212d; color: #fff; border-bottom: 4px solid #30a46c; }}
    .header-inner, main {{ width: min(1440px, calc(100% - 32px)); margin: 0 auto; }}
    .header-inner {{ min-height: 76px; display: flex; align-items: center; justify-content: space-between; gap: 24px; }}
    h1 {{ margin: 0; font-size: 22px; font-weight: 700; }}
    h2 {{ margin: 2px 0 0; font-size: 20px; }}
    h3 {{ margin: 0 0 12px; font-size: 14px; }}
    .header-meta {{ color: #c7d0d9; font-size: 12px; text-align: right; }}
    .tabs {{ background: var(--surface); border-bottom: 1px solid var(--line); position: sticky; top: 0; z-index: 5; }}
    .tab-list {{ width: min(1440px, calc(100% - 32px)); margin: 0 auto; display: flex; gap: 0; overflow-x: auto; }}
    .tab {{ min-width: 130px; height: 46px; border: 0; border-bottom: 3px solid transparent; background: transparent; color: var(--muted); font-weight: 700; cursor: pointer; }}
    .tab[aria-selected="true"] {{ color: var(--blue); border-bottom-color: var(--blue); }}
    main {{ padding: 22px 0 48px; }}
    .panel {{ display: none; }}
    .panel.active {{ display: block; }}
    .track-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; margin-bottom: 18px; }}
    .eyebrow {{ color: var(--muted); font-size: 11px; font-weight: 700; }}
    .mode {{ border: 1px solid var(--line); border-radius: 4px; padding: 4px 7px; font-size: 11px; font-weight: 700; }}
    .mode-real {{ color: var(--green); border-color: #91c7aa; background: #edf8f1; }}
    .mode-synthetic, .mode-demo {{ color: var(--amber); border-color: #d9bb78; background: #fff8e6; }}
    .mode-unknown {{ color: var(--muted); background: #f4f6f8; }}
    .notice {{ border-left: 4px solid var(--amber); background: #fff9e8; padding: 12px 14px; margin-bottom: 18px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border: 1px solid var(--line); background: var(--surface); margin-bottom: 18px; }}
    .metric {{ min-height: 88px; padding: 15px; border-right: 1px solid var(--line); }}
    .metric:last-child {{ border-right: 0; }}
    .metric-label {{ color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
    .metric-value {{ font-size: 21px; font-weight: 700; overflow-wrap: anywhere; }}
    .metric-danger .metric-value {{ color: var(--red); }}
    .section-grid, .wafer-layout {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    section {{ min-width: 0; margin-bottom: 18px; }}
    .section-grid > section, .wafer-layout > section, .chart-shell {{ background: var(--surface); border: 1px solid var(--line); padding: 16px; }}
    .table-wrap {{ overflow: auto; max-height: 360px; border: 1px solid var(--line); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th {{ position: sticky; top: 0; background: #eef1f4; color: #43505e; text-align: left; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; white-space: nowrap; }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    .empty {{ min-height: 100px; display: grid; place-items: center; color: var(--muted); border: 1px dashed var(--line); }}
    .visual-empty {{ min-height: 260px; }}
    .wafer-image {{ width: 100%; max-height: 330px; object-fit: contain; background: #fff; }}
    .chart-shell {{ height: 330px; padding: 12px; }}
    canvas {{ width: 100%; height: 100%; display: block; }}
    @media (max-width: 900px) {{
      .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .metric:nth-child(2) {{ border-right: 0; }}
      .metric:nth-child(-n+2) {{ border-bottom: 1px solid var(--line); }}
      .section-grid, .wafer-layout {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 560px) {{
      .header-inner, main, .tab-list {{ width: min(100% - 20px, 1440px); }}
      .header-inner {{ align-items: flex-start; flex-direction: column; justify-content: center; padding: 14px 0; gap: 4px; }}
      .header-meta {{ text-align: left; }}
      .metric-grid {{ grid-template-columns: 1fr; }}
      .metric {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .metric:last-child {{ border-bottom: 0; }}
    }}
  </style>
</head>
<body>
  <header><div class="header-inner"><h1>Semiconductor AI Operations</h1><div class="header-meta">Track-level integrated view<br>No row-order joins</div></div></header>
  <nav class="tabs" aria-label="Analysis views"><div class="tab-list" role="tablist">
    <button class="tab" role="tab" aria-selected="true" data-panel="overview">Overview</button>
    <button class="tab" role="tab" aria-selected="false" data-panel="secom">SECOM</button>
    <button class="tab" role="tab" aria-selected="false" data-panel="wafer">Wafer maps</button>
    <button class="tab" role="tab" aria-selected="false" data-panel="equipment">Equipment</button>
  </div></nav>
  <main>
    <div id="overview" class="panel active" role="tabpanel">
      <section class="track-head"><div><div class="eyebrow">INTEGRATED STATUS</div><h2>Manufacturing AI Overview</h2></div></section>
      <div class="notice"><strong>Data boundary:</strong> these are independent track summaries. The dashboard does not imply sample-level linkage between SECOM, wafer, and equipment records.</div>
      <div class="metric-grid">{overview_metrics}</div>
      <section><h3>Track availability and provenance</h3>{_table_html(availability_rows)}</section>
    </div>
    <div id="secom" class="panel" role="tabpanel">{secom_panel}</div>
    <div id="wafer" class="panel" role="tabpanel">{wafer_panel}</div>
    <div id="equipment" class="panel" role="tabpanel">{equipment_panel}</div>
  </main>
  <script>
    const equipmentRows = {chart_payload};
    const tabs = Array.from(document.querySelectorAll('.tab'));
    const panels = Array.from(document.querySelectorAll('.panel'));
    function activate(panelId) {{
      tabs.forEach(tab => tab.setAttribute('aria-selected', String(tab.dataset.panel === panelId)));
      panels.forEach(panel => panel.classList.toggle('active', panel.id === panelId));
      if (panelId === 'equipment') requestAnimationFrame(drawEquipmentChart);
    }}
    tabs.forEach(tab => tab.addEventListener('click', () => {{
      history.replaceState(null, '', '#' + tab.dataset.panel);
      activate(tab.dataset.panel);
    }}));
    const initialPanel = window.location.hash.slice(1);
    if (panels.some(panel => panel.id === initialPanel)) {{
      activate(initialPanel);
      requestAnimationFrame(() => window.scrollTo(0, 0));
    }}
    window.addEventListener('hashchange', () => {{
      const panelId = window.location.hash.slice(1);
      if (panels.some(panel => panel.id === panelId)) {{
        activate(panelId);
        requestAnimationFrame(() => window.scrollTo(0, 0));
      }}
    }});
    function drawEquipmentChart() {{
      const canvas = document.getElementById('equipment-chart');
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(320, Math.floor(rect.width * ratio));
      canvas.height = Math.max(220, Math.floor(rect.height * ratio));
      const ctx = canvas.getContext('2d');
      ctx.scale(ratio, ratio);
      const width = canvas.width / ratio;
      const height = canvas.height / ratio;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#647181';
      ctx.font = '12px Arial';
      if (!equipmentRows.length) {{
        ctx.fillText('No equipment anomaly scores available.', 16, 28);
        return;
      }}
      const pad = {{ left: 48, right: 18, top: 18, bottom: 34 }};
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      const scores = equipmentRows.map(row => Number(row.score) || 0);
      const min = Math.min(...scores, 0);
      const max = Math.max(...scores, 1);
      const x = index => pad.left + (index / Math.max(equipmentRows.length - 1, 1)) * plotW;
      const y = value => pad.top + (1 - (value - min) / Math.max(max - min, 0.0001)) * plotH;
      ctx.strokeStyle = '#d9dee5';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {{
        const gy = pad.top + (i / 4) * plotH;
        ctx.beginPath(); ctx.moveTo(pad.left, gy); ctx.lineTo(width - pad.right, gy); ctx.stroke();
      }}
      ctx.strokeStyle = '#1769aa';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      equipmentRows.forEach((row, index) => {{
        const px = x(index), py = y(scores[index]);
        if (index === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }});
      ctx.stroke();
      equipmentRows.forEach((row, index) => {{
        if (!row.anomaly) return;
        ctx.fillStyle = '#bd3b3b';
        ctx.beginPath(); ctx.arc(x(index), y(scores[index]), 3.5, 0, Math.PI * 2); ctx.fill();
      }});
      ctx.fillStyle = '#647181';
      ctx.fillText(max.toFixed(2), 4, pad.top + 4);
      ctx.fillText(min.toFixed(2), 4, pad.top + plotH);
      ctx.fillText('Time ordered observations', pad.left, height - 8);
    }}
    window.addEventListener('resize', () => {{
      if (document.getElementById('equipment').classList.contains('active')) drawEquipmentChart();
    }});
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return html

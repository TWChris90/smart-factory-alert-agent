"""Streamlit browser interface for the Smart Factory Alert Agent."""

from __future__ import annotations

import html
import re
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from smart_factory_alert.src.alert_agent import build_alerts
from smart_factory_alert.src.generate_data import generate_sensor_data
from smart_factory_alert.src.model_pipeline import (
    detect_with_saved_model,
    ensure_default_model,
)

FEATURES = ["temp", "pressure", "vibration"]
FONT_FAMILY = (
    "'Noto Sans TC', 'Noto Sans CJK TC', 'Microsoft JhengHei', 'PingFang TC', sans-serif"
)
SOURCE_NAMES = {
    "fixed_rule": "固定門檻",
    "knn": "組合型異常",
    # The fixed threshold is the actionable source shown to users when both
    # detectors agree; keep the combined source in the alert data itself.
    "fixed_rule+knn": "固定門檻",
}
SENSOR_CONFIG = {
    "temp": {"title": "溫度", "normal_low": 45.0, "normal_high": 50.0, "thresholds": (43.0, 52.0), "y_title": "溫度 (°C)"},
    "pressure": {"title": "壓力", "normal_low": 1.00, "normal_high": 1.05, "thresholds": (0.97, 1.08), "y_title": "壓力"},
    "vibration": {"title": "振動", "normal_low": 0.02, "normal_high": 0.04, "thresholds": (0.07,), "y_title": "振動"},
}
SENSOR_EXPLANATION = {
    "temp": {"label": "溫度", "precision": 2, "unit": " °C"},
    "pressure": {"label": "壓力", "precision": 3, "unit": ""},
    "vibration": {"label": "振動", "precision": 3, "unit": ""},
}


def _write_uploaded_file(uploaded_file) -> Path:
    """Persist an uploaded CSV to a temporary path for the existing pipeline."""
    suffix = Path(uploaded_file.name).suffix or ".csv"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        handle.write(uploaded_file.getvalue())
    finally:
        handle.close()
    return Path(handle.name)


def _run_detection(csv_path: Path):
    """Score detection data with the independent, previously fitted model."""
    model_bundle = ensure_default_model()
    df, scores, labels = detect_with_saved_model(csv_path, model_bundle)
    alerts = build_alerts(df, scores, labels)
    _add_knn_context(alerts, model_bundle)
    return (
        df,
        scores,
        labels,
        alerts,
        float(model_bundle["score_threshold"]),
        int(model_bundle["training_rows"]),
        int(model_bundle["calibration_rows"]),
    )


def _alerts_frame(alerts: list[dict]) -> pd.DataFrame:
    """Convert alert-agent output to a compact browser table."""
    return pd.DataFrame([
        {
            "時間": alert["timestamp"],
            "溫度 (°C)": alert["temp"],
            "壓力": alert["pressure"],
            "振動": alert["vibration"],
            "組合分數": alert["anomaly_score"],
            "警報來源": SOURCE_NAMES.get(alert["source"], alert["source"]),
        }
        for alert in alerts
    ])


def _add_knn_context(alerts: list[dict], model_bundle: dict) -> None:
    """Add plain-language context for KNN-only, multivariate alerts."""
    means = model_bundle["scaler"].mean_
    scales = model_bundle["scaler"].scale_
    score_threshold = float(model_bundle["score_threshold"])

    for alert in alerts:
        if alert["source"] != "knn":
            continue

        deviations = []
        for index, feature in enumerate(FEATURES):
            value = float(alert[feature])
            mean = float(means[index])
            standardized_distance = abs((value - mean) / float(scales[index]))
            deviations.append((standardized_distance, feature, value, mean))
        deviations.sort(reverse=True)

        detail_parts = []
        direction_parts = []
        sensor_labels = []
        for _, feature, value, mean in deviations[:2]:
            config = SENSOR_EXPLANATION[feature]
            direction = "偏高" if value > mean else "偏低"
            precision = config["precision"]
            unit = config["unit"]
            value_text = f"{value:.{precision}f}{unit}"
            mean_text = f"{mean:.{precision}f}{unit}"
            detail_parts.append(
                f'{config["label"]} {value_text}（正常平均 {mean_text}，{direction}）'
            )
            direction_parts.append(f'{config["label"]}{direction}')
            sensor_labels.append(config["label"])

        alert["knn_reason_zh"] = (
            "三項感測值都未超出固定異常門檻，但合併後的"
            f"主要問題：{'；'.join(detail_parts)}。"
            f'組合分數 {alert["anomaly_score"]:.4f} > 判定門檻 {score_threshold:.4f}，'
            "因此標記為需要留意的組合型異常；這不代表已確認設備故障。"
        )
        alert["knn_suggestion_zh"] = (
            f"先查看這個時間點前後 5–10 分鐘，確認{'、'.join(direction_parts)}是否持續。"
            "若只有單一時間點且後續恢復，可先記錄觀察；若連續出現，"
            f"再檢查製程設定與{'、'.join(sensor_labels)}感測器校正。"
        )


def _translate_reason(reason: str, alert: dict | None = None) -> str:
    """Translate existing alert-agent reasons without changing alert logic."""
    if reason.startswith("temperature out of range"):
        match = re.search(r"\((-?\d+(?:\.\d+)?)", reason)
        if match:
            value_text = match.group(1)
            value = float(value_text)
            operator, threshold = (">", 52.0) if value > 52.0 else ("<", 43.0)
            return f"溫度超出異常門檻 ({value_text} °C {operator} {threshold:g} °C)"
        return reason.replace("temperature out of range", "溫度超出異常門檻", 1)
    if reason.startswith("pressure out of range"):
        match = re.search(r"\((-?\d+(?:\.\d+)?)", reason)
        if match:
            value_text = match.group(1)
            value = float(value_text)
            operator, threshold = (">", 1.08) if value > 1.08 else ("<", 0.97)
            return f"壓力超出異常門檻 ({value_text} {operator} {threshold:g})"
        return reason.replace("pressure out of range", "壓力超出異常門檻", 1)
    if reason.startswith("vibration above limit"):
        match = re.search(r"\((-?\d+(?:\.\d+)?)", reason)
        if match:
            value_text = match.group(1)
            return f"振動超出異常門檻 ({value_text} > 0.07)"
        return reason.replace("vibration above limit", "振動超出異常門檻", 1)
    if reason == "PyOD KNN detected an anomalous sensor pattern.":
        if alert and "knn_reason_zh" in alert:
            return alert["knn_reason_zh"]
        return "三項感測值的組合在正常訓練資料中較少見。"
    return reason


def _translate_suggestion(suggestion: str, alert: dict | None = None) -> str:
    """Translate existing alert-agent suggestions without changing alert logic."""
    translations = {
        "Inspect cooling/heating control and verify the temperature sensor.": "檢查冷卻／加熱控制，並確認溫度感測器狀態。",
        "Check pressure regulation, valves, and pressure-sensor calibration.": "檢查壓力調節、閥件與壓力感測器校正。",
        "Inspect rotating components, mounting, and vibration sensor health.": "檢查旋轉元件、固定結構與振動感測器狀態。",
        "Inspect the equipment and review recent sensor history.": "檢查設備，並查看近期感測器歷史紀錄。",
    }
    if suggestion == "Inspect the equipment and review recent sensor history.":
        if alert and "knn_suggestion_zh" in alert:
            return alert["knn_suggestion_zh"]
    return translations.get(suggestion, suggestion)


def _fixed_rule_triggered(df: pd.DataFrame, feature: str) -> pd.Series:
    """Return rows where this specific sensor crosses its fixed threshold."""
    values = df[feature]
    if feature == "temp":
        return (values < 43.0) | (values > 52.0)
    if feature == "pressure":
        return (values < 0.97) | (values > 1.08)
    return values > 0.07


def _value_exceeds_threshold(feature: str, value: float) -> bool:
    """Return whether one sensor value crosses its fixed anomaly threshold."""
    if feature == "temp":
        return value < 43.0 or value > 52.0
    if feature == "pressure":
        return value < 0.97 or value > 1.08
    return value > 0.07


def _render_detail_metric(
    column,
    label: str,
    value: str,
    *,
    is_anomalous: bool = False,
    caption: str | None = None,
) -> None:
    """Render a detail metric with optional red anomaly emphasis and context."""
    value_class = " alert-metric__value--anomalous" if is_anomalous else ""
    caption_html = (
        f'<div class="alert-metric__caption">{html.escape(caption)}</div>'
        if caption
        else ""
    )
    column.markdown(
        f"""
        <div class="alert-metric">
            <div class="alert-metric__label">{html.escape(label)}</div>
            <div class="alert-metric__value{value_class}">{html.escape(value)}</div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sensor_charts(df: pd.DataFrame, alerts: list[dict]) -> None:
    """Render sensor-specific markers plus normal ranges and fixed thresholds."""
    alert_timestamps = pd.to_datetime([alert["timestamp"] for alert in alerts], errors="coerce")
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    chart_columns = st.columns(len(FEATURES))

    for column, feature in zip(chart_columns, FEATURES):
        config = SENSOR_CONFIG[feature]
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=timestamps, y=df[feature], mode="lines", name="感測器趨勢", line={"width": 2}
        ))
        figure.add_hrect(
            y0=config["normal_low"], y1=config["normal_high"],
            fillcolor="rgba(34, 139, 34, 0.10)", line_width=0, layer="below",
            annotation_text="正常範圍", annotation_position="top left",
        )
        for threshold in config["thresholds"]:
            figure.add_hline(
                y=threshold, line_dash="dash",
                line={"color": "red", "width": 1.5},
                annotation_text=f"異常門檻 {threshold:g}", annotation_position="bottom right"
            )

        if alerts:
            alert_mask = timestamps.isin(alert_timestamps)
            own_rule_mask = _fixed_rule_triggered(df, feature) & alert_mask
            other_alert_mask = alert_mask & ~own_rule_mask
            if own_rule_mask.any():
                figure.add_trace(go.Scatter(
                    x=timestamps[own_rule_mask], y=df.loc[own_rule_mask, feature], mode="markers",
                    name="本感測器超過門檻", marker={"color": "red", "size": 9, "symbol": "circle"},
                ))
            if other_alert_mask.any():
                figure.add_trace(go.Scatter(
                    x=timestamps[other_alert_mask], y=df.loc[other_alert_mask, feature], mode="markers",
                    name="組合型／其他警報", marker={"color": "gold", "size": 10, "symbol": "diamond"},
                ))

        figure.update_layout(
            title=config["title"], height=320, margin={"l": 10, "r": 10, "t": 55, "b": 10},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
            xaxis_title="時間", yaxis_title=config["y_title"],
            font={"family": FONT_FAMILY},
        )
        column.plotly_chart(figure, width="stretch")


def main() -> None:
    """Render the browser UI."""
    st.set_page_config(page_title="智慧工廠警告Agent", layout="wide")
    st.markdown(
        f"""
        <style>
        .stApp, .stApp * {{ font-family: {FONT_FAMILY}; }}
        .stApp .material-symbols-rounded,
        .stApp .material-icons,
        .stApp [data-testid="stIconMaterial"] {{
            font-family: "Material Symbols Rounded" !important;
        }}
        .alert-metric {{ margin: 0.25rem 0 1rem; }}
        .alert-metric__label {{ font-size: 0.875rem; font-weight: 600; margin-bottom: 0.25rem; }}
        .alert-metric__value {{ font-size: 2.25rem; line-height: 1.2; }}
        .alert-metric__value--anomalous {{ color: #ff4b4b; font-weight: 650; }}
        .alert-metric__caption {{ color: #9aa0a6; font-size: 0.8rem; margin-top: 0.35rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("智慧工廠警告Agent")

    with st.sidebar:
        st.header("操作設定")
        data_source = st.radio("資料來源", ["產生模擬資料", "上傳 CSV"])
        uploaded_file = None
        if data_source == "產生模擬資料":
            rows = st.number_input("資料筆數", min_value=100, max_value=500, value=300, step=10)
            seed = st.number_input("隨機種子", min_value=0, value=42, step=1)
        else:
            uploaded_file = st.file_uploader("感測器 CSV", type=["csv"])
        run_detection = st.button("執行異常偵測", type="primary", width="stretch")

    if run_detection:
        csv_path = None
        try:
            if data_source == "產生模擬資料":
                data = generate_sensor_data(int(rows), int(seed))
                with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
                    data.to_csv(handle.name, index=False)
                    csv_path = Path(handle.name)
            else:
                if uploaded_file is None:
                    st.error("請先上傳 CSV 檔案。")
                    return
                csv_path = _write_uploaded_file(uploaded_file)

            (
                df,
                scores,
                labels,
                alerts,
                score_threshold,
                training_rows,
                calibration_rows,
            ) = _run_detection(csv_path)
            st.session_state["result"] = {
                "df": df,
                "scores": scores,
                "labels": labels,
                "alerts": alerts,
                "score_threshold": score_threshold,
                "training_rows": training_rows,
                "calibration_rows": calibration_rows,
                "missing_summary": df.attrs.get("missing_summary"),
            }
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"無法處理感測器資料：{exc}")

    result = st.session_state.get("result")
    if result is None:
        st.info("請選擇資料來源並按下「執行異常偵測」。")
        return

    df = result["df"]
    alerts = result["alerts"]
    score_threshold = result.get("score_threshold")
    training_rows = result.get("training_rows")
    calibration_rows = result.get("calibration_rows")
    missing_summary = result.get("missing_summary")

    st.subheader("摘要")
    metric_count = 2 + int(missing_summary is not None) + int(score_threshold is not None)
    metric_cols = st.columns(metric_count)
    metric_cols[0].metric("資料筆數", len(df))
    metric_cols[1].metric("警報數量", len(alerts))
    next_metric = 2
    if missing_summary is not None:
        missing_total = int(missing_summary["total"])
        missing_rows = int(missing_summary["rows"])
        missing_by_feature = missing_summary["by_feature"]
        metric_cols[next_metric].metric("原始缺值數量", missing_total)
        if missing_total:
            metric_cols[next_metric].caption(
                f"影響 {missing_rows} 筆；溫度 {missing_by_feature['temp']}、"
                f"壓力 {missing_by_feature['pressure']}、振動 {missing_by_feature['vibration']}。"
            )
        else:
            metric_cols[next_metric].caption("未發現缺值，不需要補值")
        next_metric += 1
    if score_threshold is not None:
        metric_cols[next_metric].metric("組合判定門檻", f"{score_threshold:.4f}")
        metric_cols[next_metric].caption(
            f"三項感測值合併計算；≤ {score_threshold:.4f} 為常見組合，"
            f"> {score_threshold:.4f} 為少見組合"
        )

    st.subheader("感測器趨勢")
    _render_sensor_charts(df, alerts)

    st.subheader("警報結果（僅顯示異常時間點）")

    if not alerts:
        st.success("目前沒有偵測到警報。")
        return

    st.dataframe(_alerts_frame(alerts), width="stretch", hide_index=True)

    st.subheader("警報詳細資訊")
    for index, alert in enumerate(alerts, start=1):
        label = f'警報 {index} · {alert["timestamp"]} · {SOURCE_NAMES.get(alert["source"], alert["source"])}'
        with st.expander(label):
            detail_cols = st.columns(4)
            _render_detail_metric(
                detail_cols[0], "溫度", f'{alert["temp"]:.2f} °C',
                is_anomalous=_value_exceeds_threshold("temp", alert["temp"]),
            )
            _render_detail_metric(
                detail_cols[1], "壓力", f'{alert["pressure"]:.3f}',
                is_anomalous=_value_exceeds_threshold("pressure", alert["pressure"]),
            )
            _render_detail_metric(
                detail_cols[2], "振動", f'{alert["vibration"]:.3f}',
                is_anomalous=_value_exceeds_threshold("vibration", alert["vibration"]),
            )
            score_caption = None
            score_is_anomalous = False
            if score_threshold is not None:
                score_caption = (
                    f"常見組合 ≤ {score_threshold:.4f}；少見組合 > {score_threshold:.4f}"
                )
                score_is_anomalous = alert["anomaly_score"] > score_threshold
            _render_detail_metric(
                detail_cols[3], "組合分數", f'{alert["anomaly_score"]:.4f}',
                is_anomalous=score_is_anomalous,
                caption=score_caption,
            )
            st.markdown("**異常原因**")
            for reason in alert["reasons"]:
                st.write(f"- {_translate_reason(reason, alert)}")
            st.markdown("**建議處置**")
            for suggestion in alert["suggestions"]:
                st.write(f"- {_translate_suggestion(suggestion, alert)}")


if __name__ == "__main__":
    main()

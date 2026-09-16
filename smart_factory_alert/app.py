"""Streamlit browser interface for the Smart Factory Alert Agent."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from smart_factory_alert.src.alert_agent import build_alerts
from smart_factory_alert.src.generate_data import generate_sensor_data
from smart_factory_alert.src.preprocess import load_and_preprocess
from smart_factory_alert.src.pyod_detector import detect_anomalies

FEATURES = ["temp", "pressure", "vibration"]
SENSOR_TITLES = {
    "temp": "Temperature",
    "pressure": "Pressure",
    "vibration": "Vibration",
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


def _run_detection(csv_path: Path, contamination: float):
    """Reuse the existing preprocessing, KNN detector, and alert-agent flow."""
    df, X = load_and_preprocess(csv_path)
    scores, labels = detect_anomalies(X, contamination=contamination)
    alerts = build_alerts(df, scores, labels)
    return df, scores, labels, alerts


def _alerts_frame(alerts: list[dict]) -> pd.DataFrame:
    """Convert alert-agent output to a compact browser table."""
    return pd.DataFrame(
        [
            {
                "timestamp": alert["timestamp"],
                "temp": alert["temp"],
                "pressure": alert["pressure"],
                "vibration": alert["vibration"],
                "anomaly score": alert["anomaly_score"],
                "source": alert["source"],
            }
            for alert in alerts
        ]
    )


def _render_sensor_charts(df: pd.DataFrame, alerts: list[dict]) -> None:
    """Render each sensor trend and highlight rows that triggered an alert."""
    alert_timestamps = pd.to_datetime(
        [alert["timestamp"] for alert in alerts], errors="coerce"
    )
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    alert_mask = timestamps.isin(alert_timestamps)

    chart_columns = st.columns(len(FEATURES))
    for column, feature in zip(chart_columns, FEATURES):
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=timestamps,
                y=df[feature],
                mode="lines",
                name=SENSOR_TITLES[feature],
            )
        )

        if alert_mask.any():
            figure.add_trace(
                go.Scatter(
                    x=timestamps[alert_mask],
                    y=df.loc[alert_mask, feature],
                    mode="markers",
                    name="Alert",
                    marker={"color": "#d62728", "size": 8},
                )
            )

        figure.update_layout(
            title=SENSOR_TITLES[feature],
            height=300,
            margin={"l": 10, "r": 10, "t": 45, "b": 10},
            showlegend=False,
            xaxis_title="Timestamp",
            yaxis_title=feature,
        )
        column.plotly_chart(figure, width="stretch")


def main() -> None:
    """Render the browser UI."""
    st.set_page_config(page_title="Smart Factory Alert Agent", layout="wide")
    st.title("Smart Factory Alert Agent")
    st.caption("Sensor anomaly detection with the existing PyOD KNN pipeline")

    with st.sidebar:
        st.header("Controls")
        data_source = st.radio("Data source", ["Generate Dummy Data", "Upload CSV"])

        uploaded_file = None
        if data_source == "Generate Dummy Data":
            rows = st.number_input("Rows", min_value=100, max_value=500, value=300, step=10)
            seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        else:
            uploaded_file = st.file_uploader("Sensor CSV", type=["csv"])

        contamination = st.number_input(
            "PyOD contamination",
            min_value=0.01,
            max_value=0.50,
            value=0.05,
            step=0.01,
            format="%.2f",
        )
        run_detection = st.button("Run Detection", type="primary", use_container_width=True)

    if run_detection:
        csv_path = None
        try:
            if data_source == "Generate Dummy Data":
                data = generate_sensor_data(int(rows), int(seed))
                with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
                    data.to_csv(handle.name, index=False)
                    csv_path = Path(handle.name)
            else:
                if uploaded_file is None:
                    st.error("Please upload a CSV file first.")
                    return
                csv_path = _write_uploaded_file(uploaded_file)

            df, scores, labels, alerts = _run_detection(csv_path, float(contamination))
            st.session_state["result"] = {
                "df": df,
                "scores": scores,
                "labels": labels,
                "alerts": alerts,
                "contamination": float(contamination),
            }
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"Unable to process sensor data: {exc}")

    result = st.session_state.get("result")
    if result is None:
        st.info("Choose a data source, set contamination, and click Run Detection.")
        return

    df = result["df"]
    scores = result["scores"]
    alerts = result["alerts"]

    st.subheader("Summary")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Total Records", len(df))
    metric_cols[1].metric("Alert Count", len(alerts))
    metric_cols[2].metric("Contamination", f'{result["contamination"]:.2f}')

    st.subheader("Sensor Overview")
    _render_sensor_charts(df, alerts)

    st.subheader("Alert Results")
    if not alerts:
        st.success("No alerts detected.")
        return

    alert_table = _alerts_frame(alerts)
    st.dataframe(alert_table, width="stretch", hide_index=True)

    st.subheader("Alert Details")
    for index, alert in enumerate(alerts, start=1):
        label = f'Alert {index} · {alert["timestamp"]} · {alert["source"]}'
        with st.expander(label):
            detail_cols = st.columns(4)
            detail_cols[0].metric("Temperature", f'{alert["temp"]:.2f} °C')
            detail_cols[1].metric("Pressure", f'{alert["pressure"]:.3f}')
            detail_cols[2].metric("Vibration", f'{alert["vibration"]:.3f}')
            detail_cols[3].metric("Anomaly Score", f'{alert["anomaly_score"]:.4f}')

            st.markdown("**Reasons**")
            for reason in alert["reasons"]:
                st.write(f"- {reason}")

            st.markdown("**Suggested Actions**")
            for suggestion in alert["suggestions"]:
                st.write(f"- {suggestion}")


if __name__ == "__main__":
    main()

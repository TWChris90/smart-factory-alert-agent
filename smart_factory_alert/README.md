# Smart Factory Alert Agent

Initial implementation using PyOD for smart-factory sensor anomaly detection and alerting.

## Workflow

Sensor Data → Preprocessing → PyOD Anomaly Detection → Alert Agent → CLI / Browser UI

## Features

- Generate dummy temperature, pressure and vibration sensor data.
- Handle missing numeric values with median imputation.
- Standardize sensor features before PyOD inference.
- Detect anomalies with PyOD `KNN(n_neighbors=5)`.
- Print anomaly scores and labels.
- Generate human-readable anomaly reasons and suggested actions.

## Setup

From the repository root:

```bash
python -m pip install -r smart_factory_alert/requirements.txt
```

## CLI

```bash
python smart_factory_alert/src/cli.py --generate
python smart_factory_alert/src/cli.py --input data/sensor_data.csv
```

## Data

The generated CSV contains `timestamp`, `temp`, `pressure`, `vibration`, and `label` columns.
The PyOD model is fitted without using the `label` column.

## PyOD

The initial implementation uses `pyod.models.knn.KNN(n_neighbors=5)` on the
standardized sensor features. The `label` column is not used as a model input.
The alert agent raises an alert when either KNN or a fixed sensor threshold is
triggered. The training and detection data flow can be refined during runtime
verification based on the simplest reliable implementation.

## Browser UI

The project also provides a simple Streamlit browser interface that reuses the existing
preprocessing, PyOD KNN, and alert-agent functions. It supports generated dummy data or
CSV upload, a contamination control, summary metrics, sensor trends with highlighted alert
points, and expandable alert details.

Start the browser UI from the repository root:

```bash
streamlit run smart_factory_alert/app.py
```

The browser will open the local Streamlit page. Click **Run Detection** after choosing
a data source and contamination value.

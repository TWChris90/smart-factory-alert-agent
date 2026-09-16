# Smart Factory Alert Agent Checklist

## Planning
- [x] Inspect PyOD structure
- [x] Define basic architecture
- [x] Confirm implementation plan

## Initial Version
- [x] Generate dummy sensor data
- [x] Implement preprocessing
- [x] Implement PyOD KNN anomaly detection
- [x] Output anomaly scores and labels
- [x] Generate alert reasons and suggestions
- [x] Add CLI
- [x] Add README

## Verification
- [x] Run basic functional test
- [x] Static code review completed
- [x] Review CLI output

## Runtime Verification Results

- `python -m compileall -q smart_factory_alert/src` completed successfully.
- `python smart_factory_alert/src/generate_data.py --rows 300 --seed 42 --output data/sensor_data.csv` produced 300 data rows with the required five columns, 15 labeled anomalies, and three missing sensor values.
- Median imputation removed all feature missing values; `StandardScaler` produced three standardized features with mean approximately 0 and standard deviation 1.
- `python smart_factory_alert/src/cli.py --input data/sensor_data.csv --contamination 0.05` reported 15 alerts with anomaly scores, reasons, and suggestions.
- Both direct-script and `python -m smart_factory_alert.src.cli` entry points completed successfully.
- Controlled checks confirmed a fixed-threshold-only alert (`fixed_rule`) and a KNN-only alert (`knn`).

> Runtime verification was completed locally by Codex.

## Browser UI Revision

- [x] Keep the existing data generation, preprocessing, PyOD KNN, and alert-agent pipeline unchanged.
- [x] Convert browser-facing UI and displayed alert reasons/suggestions to Traditional Chinese at the UI layer.
- [x] Remove the browser contamination control and its Summary display.
- [x] Add per-sensor normal-range bands and fixed-threshold lines.
- [x] Render red circles only for the sensor that crosses its own fixed threshold.
- [x] Render yellow diamonds for KNN-only or other-sensor alerts when that sensor is within its fixed threshold.
- [x] Preserve existing alert source, anomaly score, reason, and suggestion data.
- [x] Verify generated-data, CSV-upload, runtime, and manual screenshot behavior.

### Browser UI Revision Results

- `python -m compileall -q smart_factory_alert/app.py` and `python -m pip check` completed successfully in the existing environment.
- The revised Streamlit app loaded in an actual local Chromium browser. The generated-data and CSV-upload flows each produced 300 records and 15 alerts; uploaded sensor values had no remaining missing values after preprocessing.
- Screenshot review confirmed readable Traditional Chinese labels, each chart's green normal-range band, black dashed threshold lines, red-circle and yellow-diamond legend semantics, and restored Streamlit sidebar icon rendering.
- Runtime checks confirmed that fixed-threshold points are red circles only on their owning sensor chart. A controlled in-range KNN-only point was rendered as a yellow diamond on all three charts and never as a red circle.
- No project files were deleted during this revision.

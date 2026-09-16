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

## Browser UI Planning
- [x] Define browser UI goal and scope
- [x] Select Streamlit as the minimal UI technology
- [x] Define data input / generation controls
- [x] Define contamination control
- [x] Define summary metrics
- [x] Define sensor trend presentation
- [x] Define alert table and alert detail presentation
- [x] Define screenshot-oriented layout
- [x] Define minimal new/modified files
- [x] Define reuse of existing pipeline without duplicating detection logic

## Browser UI Implementation
- [x] Add Streamlit `app.py`
- [x] Add Streamlit dependency to `requirements.txt`
- [x] Update README with browser UI startup instructions
- [x] Verify data generation / upload flow
- [x] Verify anomaly results and alert details in browser
- [ ] Review screenshot quality / layout

## Browser UI Runtime Verification

- [x] Confirm Streamlit starts and serves the app on localhost.
- [x] Generate dummy data through the UI and run detection.
- [x] Upload `data/sensor_data.csv` through the UI and run detection.
- [x] Confirm missing sensor values are imputed before detection.
- [x] Confirm PyOD KNN scores/labels and fixed-threshold alert sources appear in results.
- [x] Confirm Summary shows total records, alert count, and contamination.
- [x] Confirm three sensor trend charts render with alert markers.
- [x] Confirm alert table contains sensor values, scores, and source.
- [x] Confirm alert details contain reasons and suggested actions.
- [ ] Confirm screenshot quality and visual layout manually.

### Commands and Results

- `.venv/bin/python -m compileall -q smart_factory_alert` completed successfully.
- `.venv/bin/python -m pip check` reported no broken requirements.
- `.venv/bin/python -m streamlit run smart_factory_alert/app.py --server.headless true --server.address 127.0.0.1 --server.port 8501` started successfully; `curl http://127.0.0.1:8501/` returned the Streamlit page.
- Streamlit `AppTest` verified both generated-data and CSV-upload paths with no app exceptions: 300 records, 15 alerts, three charts, one six-column alert table, and 15 alert detail expanders.
- Uploaded-data verification confirmed zero missing values in `temp`, `pressure`, and `vibration` after preprocessing, with 300 scores and 300 labels.

> Manual screenshot/visual-quality review was not performed because no browser screenshot inspection tool was available; that item remains unchecked.

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

## Browser UI Revision
- [x] Convert browser-facing UI and displayed alert reasons/suggestions to Traditional Chinese at the UI layer.
- [x] Remove the sidebar PyOD contamination input.
- [x] Remove Contamination from the Summary and all other general-user UI areas.
- [x] Reuse the existing KNN pipeline/default setting without changing KNN or alert logic.
- [x] Show each sensor's normal range and fixed anomaly threshold(s) on its own chart.
- [x] Use red circles only when that sensor itself crosses its fixed threshold.
- [x] Use yellow diamonds for KNN-only or other-source alerts when that sensor itself does not cross its fixed threshold.
- [x] Preserve existing alert source, anomaly score, reason, and suggestion data in the alert results/details.
- [x] Update README for the revised browser UI behavior.
- [x] Complete static code review.

## Browser UI Revision Verification
- [x] Run the revised Streamlit app and verify the browser renders correctly.
- [x] Verify the revised generated-data and CSV-upload flows in the browser.
- [x] Verify contamination is absent from the sidebar, Summary, and other user-facing UI.
- [x] Verify all browser-facing labels and alert explanations display in Traditional Chinese.
- [x] Verify each sensor chart shows its normal range and fixed threshold(s).
- [x] Verify red circles appear only for the sensor that itself crosses its fixed threshold.
- [x] Verify yellow diamonds represent KNN-only or other-source alerts without implying that sensor itself is abnormal.
- [x] Verify alert table/details still show the existing sources, scores, reasons, and suggestions.
- [x] Perform manual screenshot / visual-quality review.

### Browser UI Revision Commands and Results

- `.venv/bin/python -m compileall -q smart_factory_alert/app.py` and `.venv/bin/python -m pip check` completed successfully.
- `.venv/bin/python -m streamlit run smart_factory_alert/app.py --server.headless true --server.address 127.0.0.1 --server.port 8501` started successfully; an actual Chromium browser session loaded and operated the app.
- Generated-data and CSV-upload UI flows each completed without app exceptions: 300 records, 15 alerts, three sensor charts, one alert table, and 15 detail expanders. Uploaded sensor features had zero missing values after preprocessing.
- Actual browser screenshots confirmed readable Traditional Chinese UI text, a green normal-range band, black dashed fixed-threshold line(s), and clear legends on all three charts.
- Generated data produced five red circles on the owning sensor chart and ten yellow diamonds on the other sensor charts. A controlled in-range KNN-only candidate was `source=knn`, appeared as a yellow diamond in all three charts, and was never drawn as a red circle.
- The actual browser alert-detail view preserved source and score, and displayed translated reasons and suggested actions.
- The global UI font fallback was adjusted in `app.py` to render Traditional Chinese text without replacing Streamlit Material icons. No files were deleted.

> Browser UI Revision runtime and visual verification was completed locally by Codex.

## Model Training / Inference Separation Revision

### Planning

- [ ] Define a separate normal-only training dataset.
- [ ] Define a strict boundary between KNN training data and detection data.
- [ ] Define saved KNN model and preprocessing state artifacts.
- [ ] Define training-time KNN score threshold determination.
- [ ] Define inference-only flow for generated and uploaded data.
- [ ] Define seed-based randomized anomaly and missing-value placement.
- [ ] Preserve the existing KNN algorithm, fixed thresholds, alert logic, CLI, and Browser UI.

### Implementation

- [ ] Separate KNN training / fit from detection / inference.
- [ ] Train KNN only on independent normal data.
- [ ] Save the fitted KNN model.
- [ ] Save the preprocessing state required for inference.
- [ ] Determine and save the KNN score threshold from training data only.
- [ ] Prevent generated / uploaded detection data from being used to fit KNN.
- [ ] Remove any assumption that the first N rows are normal.
- [ ] Randomize anomaly placement/type and missing-value placement using the random seed.
- [ ] Ensure the same seed is reproducible and different seeds can produce different distributions.
- [ ] Preserve current fixed-threshold and alert-agent logic.
- [ ] Preserve the current Browser UI behavior.
- [ ] Update README with the training / inference lifecycle.

### Verification

- [ ] Verify the saved KNN model can be loaded for inference without fitting on detection data.
- [ ] Verify the KNN score threshold comes only from independent normal training data.
- [ ] Verify an uploaded/generated detection dataset does not change the saved model or threshold.
- [ ] Verify detection does not assume the first rows are normal.
- [ ] Verify same random seed reproduces anomaly and missing-value distribution.
- [ ] Verify different random seeds can produce different anomaly and missing-value distributions.
- [ ] Verify fixed-threshold alerts remain unchanged.
- [ ] Verify KNN-only alerts remain available with the existing alert source semantics.
- [ ] Verify current CLI behavior remains compatible.
- [ ] Verify current Browser UI remains compatible.
- [ ] Run functional tests only after implementation is confirmed.
- [ ] Perform static code review.

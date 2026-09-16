# Smart Factory Alert Agent — Browser UI Revision Plan

## Goal

在既有智慧工廠感測器異常偵測與瀏覽器介面上，針對實際操作發現的可用性問題做最小 UI 修改：

1. 將瀏覽器介面與警報內容改為繁體中文。
2. 在三張感測器圖表上清楚呈現正常範圍與固定異常門檻。
3. 不讓一般使用者操作 PyOD `contamination`；沿用現有 KNN / alert pipeline 的既有設定，不改模型流程。
4. 避免三張圖在同一警報時間全部顯示紅點；每張圖只標示該感測器實際觸發固定門檻的資料點。

本階段只修改瀏覽器介面與相關說明文件，不修改資料生成、前處理、PyOD KNN、警告判斷邏輯或資料格式。

## Existing Workflow

```text
Sensor Data
→ Preprocessing
→ PyOD KNN Anomaly Detection
→ Alert Agent
→ CLI / Browser UI
```

既有核心流程已完成並通過本機驗證；本階段維持不變。

## Minimal UI Changes

### 1. Chinese UI

將 `app.py` 中面向一般使用者的英文文字改為繁體中文，包括：

- page title / title / caption
- sidebar controls
- data source choices
- rows / seed labels
- Run Detection
- Summary metrics
- Sensor Overview
- Alert Results / Alert Details
- table column names
- Reasons / Suggested Actions
- no-alert / input / processing messages

警報內容沿用既有 `alert_agent.py` 的判斷結果；本階段只在 UI 層將既有 reason / suggestion 的展示文字改為繁體中文。不得改變 alert 判斷條件。

### 2. Sensor Chart Ranges and Thresholds

三張圖各自呈現自己的感測器規格，不共用一組紅點：

| Sensor | 正常範圍 | 固定異常門檻 |
|---|---|---|
| Temperature | 45–50 °C | `< 43 °C` 或 `> 52 °C` |
| Pressure | 1.00–1.05 | `< 0.97` 或 `> 1.08` |
| Vibration | 0.02–0.04 | `> 0.07` |

圖表至少要能辨識：

- sensor trend line
- normal range band
- fixed threshold line(s)
- 該 sensor 自己實際觸發固定門檻的資料點

正常範圍與固定門檻只做視覺化，不新增任何判斷邏輯。

### 3. Hide Contamination from General Users

移除瀏覽器側邊欄的 `PyOD contamination` 操作元件；同時移除 Summary 中的 `Contamination` 顯示。

UI 不新增替代模型參數控制，也不得在其他一般使用者可見區域顯示此參數；使用既有 pipeline 的預設 / 既有 contamination 設定直接執行 KNN。

不得修改 `pyod_detector.py`、KNN `n_neighbors=5`、alert source 判斷或其他核心模型流程。

### 4. Sensor-specific Alert Markers

目前畫面使用「所有 alert timestamp」作為所有圖表的紅點來源，會讓一個 KNN-only alert 在三張圖上看起來都像是三個 sensor 同時異常。

最小修改方式：

- Temperature 圖：該 sensor 自己超過固定門檻時，以**紅色圓點**標示。
- Pressure 圖：該 sensor 自己超過固定門檻時，以**紅色圓點**標示。
- Vibration 圖：該 sensor 自己超過固定門檻時，以**紅色圓點**標示。
- KNN-only 或其他感測器觸發的 alert，若該 sensor 自己沒有超過固定門檻，則以**黃色菱形**表示該時間點存在其他來源的 alert，而不是紅色圓點。
- 純 KNN-only alert 若沒有對應的 sensor 固定門檻觸發，不在任一單一 sensor 圖上假裝成該 sensor 的紅點。
- Alert table / Alert Details 仍保留既有 KNN / fixed_rule / combined alert source、score、reason、suggestion 資訊。

這項修改只改圖表呈現，不改 `build_alerts()` 的告警判斷。

## Planned File Changes

只預計修改：

```text
smart_factory_alert/
├── app.py          # UI wording, chart annotations, sensor-specific markers, hide contamination control
├── README.md       # 更新瀏覽器操作說明與 UI 行為
├── plan.md         # 本規劃
└── checklist.md    # 本階段 checklist
```

不修改：

```text
smart_factory_alert/src/generate_data.py
smart_factory_alert/src/preprocess.py
smart_factory_alert/src/pyod_detector.py
smart_factory_alert/src/alert_agent.py
smart_factory_alert/src/cli.py
smart_factory_alert/data/*
```

## Non-goals

本階段不加入：

- 新模型 / 新 anomaly detector
- contamination slider / input replacement
- 新的 alert 判斷條件
- database / history
- real-time streaming
- notification service
- authentication
- advanced agent framework
- 額外 dashboard / analytics

## Implementation Order

1. 將 `app.py` 的使用者可見文字改為繁體中文。
2. 隱藏 contamination 操作，改用既有預設 / pipeline 設定。
3. 在三張圖加入各自的正常範圍與固定異常門檻線 / 區域。
4. 將紅點來源改成 sensor-specific fixed-threshold trigger；保留既有 alert table/detail。
5. 更新 README 的瀏覽器操作說明。
6. 完成後只做靜態 code review；本階段不測試、不 compile。

## Browser UI Revision — Implementation Status

- Implemented the confirmed UI-only revision in `app.py`.
- Updated `README.md` to document the revised browser behavior.
- Existing data generation, preprocessing, PyOD KNN, CLI, and alert-agent logic were left unchanged.
- Completed static, runtime, browser, and screenshot verification for the revised UI.
- Verified the normal-range bands, fixed-threshold lines, red-circle markers, and yellow-diamond markers against generated data and a controlled KNN-only case.
- No files were deleted for this revision.

## Independent Training and Persistent Model Revision

The KNN workflow now separates training from detection:

1. Generate an independent normal-only baseline with its own fixed training seed.
2. Fit the imputer, scaler, and KNN only on that baseline.
3. Save those fitted objects and the learned score threshold in one joblib model.
4. Load the saved model for all generated or uploaded detection data.
5. Apply only `transform` and `decision_function` during detection; never fit on detection rows and never assume an initial normal segment.

Simulated detection data keeps one-minute timestamps while anomaly positions, anomaly types, anomaly values, and missing values are controlled by the user-selected random seed.

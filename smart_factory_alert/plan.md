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

## Model Training / Inference Separation Revision

### Goal

修正目前「每次偵測都使用待測資料重新 fit KNN」造成的資料洩漏問題，將 KNN 流程分成獨立的訓練階段與偵測階段：

```text
Independent Normal Training Data
→ Preprocessing / Feature Transformation
→ KNN Training
→ Determine KNN Score Threshold
→ Save Model + Preprocessing State
                         ↓
New Generated / Uploaded Data
→ Load Saved Model + Preprocessing State
→ Detection Only
→ Existing Fixed-Threshold + KNN Alert Agent
→ Current CLI / Browser UI
```

### Required Changes

1. **獨立正常訓練資料**
   - 建立與待測資料分離的正常資料集，僅供 KNN 建模。
   - 訓練資料必須不含人工注入的異常樣本。
   - 不以「資料前幾筆一定正常」作為任何訓練資料假設。
   - 訓練資料的取得／建立方式需能由 random seed 重現。

2. **KNN Training**
   - 保留目前 PyOD `KNN` 演算法與既有 KNN 設定。
   - KNN 只在獨立正常訓練資料上 `fit`。
   - 由訓練資料取得 KNN anomaly score threshold。
   - 保存可供後續 inference 使用的模型與必要 preprocessing state。

3. **Detection / Inference**
   - 上傳或產生的新資料只允許載入已保存的模型與 preprocessing state 進行偵測。
   - 待測資料不得再次 `fit` KNN，也不得參與模型 threshold 建立。
   - 不假設待測資料的前幾筆為正常。
   - 保留現有 anomaly score、KNN label、固定門檻與 alert source 判斷方式。

4. **Randomized Dummy Data**
   - 異常值的位置、異常類型／sensor 分布與缺失值位置依 random seed 隨機產生。
   - 不再使用固定的異常排列方式作為隱含資料結構。
   - 相同 seed 必須可重現相同資料；不同 seed 應可得到不同但合理的異常／缺值分布。
   - 正常資料與待測資料的產生都不得依賴「前 N 筆為正常」的假設。

5. **Preserve Existing UI and Alert Behavior**
   - 保留目前 Browser UI 與 CLI 的操作方式。
   - 保留目前三張感測器圖、固定門檻視覺化，以及紅色圓點／黃色菱形語意。
   - 保留 fixed-threshold 與 KNN alert source 的既有判斷邏輯；本 revision 只改變 KNN 的訓練／載入生命週期與資料邊界。
   - 不新增 contamination 操作介面。

### Planned File Changes

依後續實作需要，僅規劃修改與 KNN lifecycle、資料產生／前處理整合及現有介面說明直接相關的檔案；不擴大到新的模型或新的 UI 功能。

預期涉及：

```text
smart_factory_alert/
├── src/generate_data.py        # 隨機異常／缺值分布，保留 seed 可重現性
├── src/preprocess.py           # 讓 training / inference 共用保存的 preprocessing state
├── src/pyod_detector.py        # 分離 train / save 與 load / detect；保留 KNN
├── src/alert_agent.py          # 不改固定門檻與 alert 判斷邏輯；僅維持既有介面相容
├── app.py                      # 沿用目前 UI；偵測改走保存模型的 inference flow
├── README.md                   # 更新 training / inference 使用方式
├── plan.md
└── checklist.md
```

不加入新模型、contamination 控制、額外 dashboard、real-time streaming 或其他 plan 外改善。

### Non-goals

- 不更換 PyOD KNN。
- 不重新設計固定門檻。
- 不改變既有 fixed-threshold / KNN alert 判斷邏輯。
- 不讓待測資料參與 KNN training 或 threshold determination。
- 不以資料前幾筆正常作為訓練策略。
- 不加入新的 UI 功能。

### Implementation Order

1. 定義獨立正常 training data 與 inference data 的資料邊界。
2. 將 KNN `fit` 與 inference `detect` 分離。
3. 在 training phase 決定並保存 KNN score threshold，以及必要 preprocessing state。
4. 修改 dummy-data randomization，使異常與缺值分布由 seed 決定且不同 seed 可產生不同分布。
5. 讓 CSV upload / generated data 只載入已保存模型進行 detection。
6. 保留現有 fixed-threshold / alert-agent 與目前 Browser UI。
7. 更新 README 說明 training / inference 流程。
8. 實作完成後再依 checklist 執行驗證；本次規劃階段不改碼、不測試。

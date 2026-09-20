# Smart Factory Alert Agent

智慧工廠感測器異常偵測警告系統。

## Dummy data

存放於 
```bash
data/sensor_data.csv
```

## Setup

在 repository root 執行：

```bash
python -m pip install -r requirements.txt
```


## Browser UI

啟動瀏覽器介面：

```bash
streamlit run app.py
```
或是
```bash
streamlit run app.py --server.headless true --server.address 0.0.0.0
```

瀏覽器介面支援兩種資料來源：

- **產生模擬資料**：可設定資料筆數與隨機種子。
- **上傳 CSV**：使用既有 CSV 資料流程進行偵測。

### 畫面內容

- **摘要**：顯示資料筆數與警報數量。
- **資料品質**：在補值前統計原始缺值總數、受影響筆數與各感測器缺值數，再使用正常訓練資料所建立的中位數補值器處理。
- **感測器趨勢**：分別顯示溫度、壓力、振動趨勢，以及各自的正常範圍與固定異常門檻。
- **警報結果**：顯示時間、感測器數值、異常分數與警報來源。
- **警報詳細資訊**：顯示既有警告流程產生的原因與建議處置。
- **組合型異常**：單項數值可能都未越界，但多個感測值同時出現的組合與正常訓練資料距離較遠。並會列出相較正常平均值偏高／偏低的主要感測器，以及單點觀察與連續異常處置。

### 感測器圖表標記

- **紅色圓點**：該感測器本身超過固定異常門檻。
- **黃色菱形**：該時間點存在其他來源的警報，該感測器本身沒有超過固定門檻。

## Data

CSV 必須包含：`timestamp`, `temp`, `pressure`, `vibration`；`label` 為選填。

## PyOD

實作使用 `pyod.models.knn.KNN(n_neighbors=5)`。補值器、標準化器與 KNN 僅使用 400 筆獨立正常訓練資料執行 `fit`；另外使用 100 筆正常校準資料的異常分數第 99.5 百分位決定門檻。補值器、標準化器、KNN 與門檻會以 joblib 一起保存。待測資料只執行保存前處理器的 `transform` 與 KNN 的 `decision_function`，不參與訓練或門檻校準。

## Workflow

```text
400 筆獨立正常訓練資料
→ fit 中位數補值器與 StandardScaler
→ fit PyOD KNN

100 筆獨立正常校準資料
→ 使用已完成的補值器、StandardScaler 與 KNN 計算分數
→ 取第 99.5 百分位作為分數門檻
→ 保存補值器、標準化器、KNN 與門檻

上傳／模擬待測資料
→ 載入保存模型
→ transform + decision_function（不重新 fit）
→ Alert Agent → CLI / Browser UI
```

## Features

- 依 random seed 隨機產生異常位置、異常類型、異常值與缺值。
- 使用 400 筆獨立正常資料訓練補值器、標準化器與 KNN。
- 使用另外 100 筆獨立正常資料校準分數門檻。
- 上傳或產生的待測資料只做轉換與推論，不參與模型訓練。
- 使用中位數補值處理缺失數值。
- 在 PyOD KNN 推論前進行感測器特徵標準化。
- 使用 PyOD `KNN(n_neighbors=5)` 偵測異常。
- 輸出 anomaly score 與異常標籤。
- 根據既有告警流程提供異常原因與建議處置。
- 提供 Streamlit 瀏覽器介面，方便資料載入、異常結果瀏覽與截圖展示。

## CLI

### 終端機 Demo 截圖

在 repository root 執行以下一行：

```bash
python -m smart_factory_alert.src.cli --input data/sensor_data.csv --limit 25 --show-normal --output outputs/demo_alerts.csv --results-output outputs/demo_results.csv
```

此指令會在同一個終端畫面顯示：載入資料筆數、缺值數、前三筆原始資料、保存模型與判定門檻、前 25 筆正常／警報結果、警報原因、建議處置及最終統計。選擇 25 筆是為了讓預設資料中的警報與第一筆缺值都能出現在截圖中，同時避免輸出過長。

結果會另外保存為：

- `outputs/demo_alerts.csv`：前 25 筆資料中的警報。
- `outputs/demo_results.csv`：前 25 筆資料的完整判定結果。

### 其他 CLI 指令

```bash
python -m smart_factory_alert.src.train_model
python smart_factory_alert/src/cli.py --generate
python smart_factory_alert/src/cli.py --input data/sensor_data.csv
```

訓練指令預設產生 400 筆獨立正常訓練資料至
`smart_factory_alert/data/normal_training_data.csv`，以及 100 筆獨立正常校準資料至
`smart_factory_alert/data/normal_calibration_data.csv`。校準資料只用來計算正常分數的第 99.5 百分位門檻，不會拿來訓練模型。模型保存至
`smart_factory_alert/models/knn_normal_model.joblib`。若預設模型不存在，CLI 與瀏覽器介面會先建立這組獨立模型；模型存在後，每次偵測只會載入，不會重新訓練。
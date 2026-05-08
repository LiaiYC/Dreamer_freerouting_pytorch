# DreamerV3 x Freerouting (PyTorch/JAX)

將 DreamerV3 強化學習框架整合到 Freerouting 自動佈線器，目標是學習一組可泛化的路由參數，提升不同 PCB (`.dsn`) 的佈線品質。

本專案不是直接預測每條走線，而是把「Freerouting 參數調校」建模成強化學習決策問題。

> **專利申請中聲明**  
> 本專案部分「神經網路訓練流程、模型設定與關鍵超參數」涉及正在申請中的專利，相關實作細節目前暫不公開，將於專利程序完成後延後開放。

## 專案重點

- 以 `DSN` 板級特徵作為觀測，透過 DreamerV3 學習路由參數。
- 環境會呼叫 Freerouting CLI 自動佈線並解析 `.ses` 與 DRC JSON。
- 支援兩種工作模式：
  - `jax`: 完整 DreamerV3 訓練流程（部分內容暫不公開）。
  - `pytorch`: 快速 benchmark 與 RL probe（部分內容暫不公開）。
- 提供資料前處理腳本，可先建立 manifest (`.jsonl`) 加速實驗。

## 目錄結構

```text
Dreamer_freerouting_pytorch/
├─ dreamerv3/                 # DreamerV3 主程式與整合腳本
│  ├─ dreamerv3/              # agent / configs / main
│  ├─ embodied/               # env 與訓練基礎框架
│  │  └─ envs/freerouting.py  # Freerouting RL 環境
│  └─ scripts/                # 訓練、benchmark、資料準備腳本
├─ freerouting/               # Freerouting 原始碼（Java）
├─ DSN/                       # PCB 測試資料（.dsn）
├─ logdir/                    # 訓練與實驗輸出
├─ .upstream_dreamerv3/       # 上游 DreamerV3 參考快照
└─ chapter_*.md/.tex          # 研究與設計文件
```

## 方法摘要

### 1) Observation（狀態）

`embodied/envs/freerouting.py` 會把每塊板子轉成 8 維特徵向量（`log1p` 壓縮）：

- 檔案大小
- nets / components / pins 數量
- layer / keepout 數量
- 板寬與板高（mm）

### 2) Action（動作）

> 此區涉及專利申請內容，以下僅提供降階描述（細節延後開放）。

策略會輸出一組「路由器控制向量」，由環境轉換成 Freerouting 可接受的設定值，用於影響佈線行為（如探索強度、繞線偏好與收斂策略）。目前不公開參數名稱、範圍映射與約束細節。

### 3) Reward（回饋）

> 此區涉及專利申請內容，以下僅提供降階描述（細節延後開放）。

回饋函數採多目標品質評估設計，會綜合「佈線可完成性、設計規則遵循程度、佈線代價」等訊號形成訓練回饋。目前不公開各項權重、懲罰/獎勵組合方式與例外處理邏輯。

## DSN 資料說明（訓練/測試）

`DSN/` 目錄中的 `.dsn` 檔案為本專案的板級資料來源，會用於：

- 訓練資料（Training Data）
  - 提供模型在訓練過程中學習不同板型與佈線情境。
- 測試資料（Testing Data）
  - 用於驗證訓練後模型在未見或保留板子上的泛化表現。

建議做法是將 `DSN/` 先切分為固定的訓練/測試子集合（例如 8:2），並在實驗中固定 seed 與資料清單，以確保結果可重現與可比較。

## 環境需求

- Python `3.11+`
- Java Runtime（建議 21）
- Freerouting `.jar`（可使用 release 版本）

## 安裝

在專案根目錄執行：

```bash
cd dreamerv3
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
```

### JAX 完整訓練依賴

```bash
pip install -U -r requirements.txt
```

### PyTorch benchmark / probe 依賴

```bash
pip install -U -r requirements-pytorch.txt
```

## 快速開始

### A. 建立資料 manifest（可選，但推薦）

```bash
python scripts/prepare_freerouting_data.py \
  --input-dir ../DSN \
  --output data/freerouting/boards.jsonl \
  --patterns "*.dsn,*.DSN" \
  --shuffle --seed 0
```

### B. PyTorch 快速 benchmark

> 此段訓練流程涉及專利申請內容，暫不公開（延後開放）。

### C. GPU/輸出檢查

> 此段訓練驗證流程涉及專利申請內容，暫不公開（延後開放）。

### D. PyTorch RL probe（真實呼叫 Freerouting）

> 此段訓練互動流程涉及專利申請內容，暫不公開（延後開放）。

### E. JAX DreamerV3 正式訓練

> 此段正式訓練流程涉及專利申請內容，暫不公開（延後開放）。

## 重要設定（`dreamerv3/configs.yaml`）

以下與神經網路訓練策略、關鍵超參數相關之細節，因專利申請中暫不公開（延後開放）。

目前公開內容以資料前處理與環境整合說明為主。

## 申請完成後開放範圍（預告清單）

以下內容將於專利申請程序完成後，依版本逐步開放：

1. Action 參數完整定義
2. Action 正規化與反映射公式（含邊界/裁切規則）
3. Reward 完整數學式與各子項權重
4. Reward shaping 策略與失敗案例處理規則
5. 訓練流程指令（JAX / PyTorch）與可重現實驗設定
6. 關鍵超參數與 ablation 設定
7. 代表性實驗結果與對照分析腳本

## 輸出與監控

- 預設輸出於 `logdir/...`
- 指標包含：
  - 任務品質：`log/length_mm`, `log/vias`, `log/violations*`, `log/unconnected`, `log/completion`
  - 學習指標：`train-loss-*`, `report-*`
  - 系統資源：CPU/RAM/GPU

可用 Scope 檢視：

```bash
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
```

## 常見問題

- `Missing --jar or FREEROUTING_JAR`
  - JAX 訓練或 RL probe 必須提供 Freerouting jar。
- `No DSN boards found...`
  - 檢查 `--data-dir` / `--manifest` 路徑與副檔名是否為 `.dsn/.DSN`。
- 指定 `--device gpu` 但實際跑在 CPU
  - 先用 `verify_torch_benchmark.py` 檢查 `resolved_device` 與 `cuda_available`。

## 備註

- `logdir/`、大型 `.ses/.json` 輸出通常不建議直接提交到 Git。
- `freerouting/` 與 `.upstream_dreamerv3/` 屬於上游/第三方程式，整合邏輯主要在 `dreamerv3/embodied/envs/freerouting.py` 與 `dreamerv3/scripts/`。

## 致謝

- DreamerV3: https://github.com/danijar/dreamerv3
- Freerouting: https://github.com/freerouting/freerouting

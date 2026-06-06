# SketchSense

這是一個模仿 Google Quick, Draw! 的深度學習期末專案。系統會從
`labels.txt` 隨機選出題目，使用者在網頁畫布上作畫時，模型會持續進行即時辨識，
直到猜中正確答案。

## 專案功能

- 從 10 個模型類別中隨機出題
- 支援滑鼠、觸控板與觸控螢幕作畫
- 停筆約 0.4 秒後自動更新辨識結果
- 顯示前三名預測類別與信心分數
- 猜中後可直接開始下一題
- 提供清除畫布、調整筆刷與更換題目功能
- 響應式網頁介面，可在電腦與平板使用

## 模型類別

| 編號 | 類別 | 編號 | 類別 |
| --- | --- | --- | --- |
| 1 | 飛機 | 6 | 手電筒 |
| 2 | 香蕉 | 7 | 吉他 |
| 3 | 相機 | 8 | 安全帽 |
| 4 | 門 | 9 | 冰淇淋 |
| 5 | 眼睛 | 10 | 夾克 |

## 使用技術

### Frontend

- React
- Vite
- HTML Canvas API
- CSS

### Backend

- Python
- FastAPI
- PyTorch
- Torchvision
- Pillow

### Model

- MobileNetV2
- 輸入格式：`28 x 28` 灰階圖片
- 輸出類別：10 類
- 模型參數量：約 224 萬

## 專案架構

```text
deeplearn_finalproject/
├── backend/
│   ├── __init__.py
│   └── app.py              # FastAPI 與模型推論
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # 畫布、遊戲流程與 API 呼叫
│   │   ├── main.jsx        # React 進入點
│   │   └── styles.css      # 網頁樣式
│   ├── .env.example
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── labels.txt              # 模型類別
├── model.pt                # PyTorch 模型權重
├── main.py                 # 後端啟動入口
├── requirements.txt        # Python 套件
└── README.md
```

## 執行環境

- Python 3.10 或 3.11
- Node.js 20 以上

## 安裝與啟動

### 1. 下載專案

```bash
git clone <your-repository-url>
cd deeplearn_finalproject
```

### 2. 啟動後端

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Windows PowerShell 啟用虛擬環境：

```powershell
.venv\Scripts\Activate.ps1
```

後端預設網址：

- API：<http://localhost:8000>
- API 文件：<http://localhost:8000/docs>

### 3. 啟動前端

開啟另一個終端機：

```bash
cd frontend
npm install
npm run dev
```

開啟 <http://localhost:5173> 即可開始使用。

## 操作方式

1. 網頁會從 `labels.txt` 隨機選出一個題目。
2. 在畫布上畫出指定物件。
3. 每次停筆後，前端會自動將畫布傳送到 FastAPI。
4. 後端將圖片轉為灰階、反相並縮放成 `28 x 28`。
5. PyTorch 模型回傳前三名預測結果。
6. 第一名預測與題目相同時，即完成該回合。

快捷鍵：

- `Ctrl/Command + Enter`：立即辨識或進入下一題
- `Esc`：清除畫布

## API

### `GET /api/health`

確認模型是否完成載入，並取得 `labels.txt` 中的所有類別。

### `POST /api/predict`

接收 Base64 PNG 圖片，回傳前三名預測與推論時間。

回傳格式：

```json
{
  "predictions": [
    {
      "label": "飛機",
      "confidence": 92.5
    }
  ],
  "latency_ms": 18.4
}
```

## 注意事項

- 前端與後端需要同時執行。
- 請勿直接開啟 `frontend/index.html`，必須使用 `npm run dev`。
- 若後端網址不是 `http://localhost:8000`，請複製 `.env.example` 為 `.env`
  並修改 `VITE_API_URL`。

## 專案目的

本專案將深度學習影像分類模型整合進完整 Web 應用，展示從使用者繪圖、
影像前處理、API 傳輸到模型即時推論的完整流程。

# Yuntech 114-2 神經網路與深度學習期末專案

這是一個模仿 Google Quick, Draw! 的深度學習期末專案。系統會從
`labels.txt` 隨機選出題目，使用者在網頁畫布上作畫時，模型會持續進行即時辨識，
直到猜中正確答案。

## 專案功能

- 從 20 個模型類別中隨機出題
- 支援滑鼠、觸控板與觸控螢幕作畫
- 停筆約 0.4 秒後自動更新辨識結果
- 顯示前三名預測類別與信心分數
- 使用 Grad-CAM 顯示模型做出判斷時關注的筆跡區域
- 猜中後可直接開始下一題
- 提供清除畫布、調整筆刷與更換題目功能
- 響應式網頁介面，可在電腦與平板使用

## 模型類別

| 編號 | 類別 | 編號 | 類別 |
| --- | --- | --- | --- |
| 1 | 飛機 | 11 | 眼睛 |
| 2 | 蘋果 | 12 | 手電筒 |
| 3 | 香蕉 | 13 | 吉他 |
| 4 | 自行車 | 14 | 帽子 |
| 5 | 鳥 | 15 | 安全帽 |
| 6 | 公車 | 16 | 冰淇淋 |
| 7 | 相機 | 17 | 夾克 |
| 8 | 貓 | 18 | 山 |
| 9 | 椅子 | 19 | 星星 |
| 10 | 門 | 20 | 雨傘 |

`labels.txt` 的行序必須與模型訓練時的類別索引一致；中文名稱只用於介面顯示，
不會改變模型的輸出順序。

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
- 輸出類別：20 類
- 模型參數量：約 224 萬

## 專案架構

```text
quickdraw-ai-demo/
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
├── training/
│   ├── 114_2神經網路與深度學習_期末專案.ipynb
│   └── README.md                  # 訓練相關檔案說明
├── labels.txt                     # 20 個中文模型類別
├── quickdraw_20_classes.pt        # 20 類 PyTorch 模型權重
├── main.py                        # 後端啟動入口
├── requirements.txt               # Python 套件
└── README.md
```

## 執行環境

- Python 3.10 或 3.11
- Node.js 20 以上

## 安裝與啟動

### 1. 下載專案

```bash
git clone <your-repository-url>
cd quickdraw-ai-demo
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
6. 系統對第一名預測反向傳播，產生 Grad-CAM 熱力圖。
7. 第一名預測與題目相同時，即完成該回合。

## 模型可解釋性

後端會針對每次第一名預測計算 Grad-CAM，而非使用固定或隨機的視覺效果。
系統取得 MobileNetV2 `features[1]` 的卷積特徵，將目標類別分數反向傳播至
該層，以梯度的全域平均作為各通道權重，最後加權、ReLU、正規化並上採樣至
原始畫布大小。為了避免低權重背景訊號干擾閱讀，視覺化時只顯示正規化後
高於 0.20 的正向類別貢獻。

由於模型輸入只有 `28 x 28`，最後卷積層的空間大小已縮小為 `1 x 1`，無法提供
有效定位資訊，因此本專案採用仍保有 `14 x 14` 空間網格的卷積特徵層。低於
關注門檻的區域會保持透明，避免低權重背景訊號被誤解為模型關注區域。

此外，系統使用輸入筆跡向外 3 個模型像素的柔性支援區域，抑制距離筆跡過遠的
背景 activation。這項後處理不會產生新的關注位置，原始熱度仍由類別梯度決定；
它只移除不利於塗鴉解讀的遠端空白雜訊，因此介面標示為 Ink-guided Grad-CAM。

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

# 模型訓練

此目錄保存 Quick, Draw! 20 類塗鴉辨識模型的訓練 notebook 與相關實驗檔案。

## 目前檔案

- `114_2神經網路與深度學習_期末專案.ipynb`：資料下載、前處理、模型訓練、
  驗證、結果視覺化與權重輸出的完整流程。

## 訓練設定

- 資料來源：Google Quick, Draw! Numpy Bitmap Dataset
- 類別數量：20 類
- 每類樣本：15,000 筆
- 總樣本數：300,000 筆
- 資料切分：90% 訓練、10% 驗證
- 模型架構：MobileNetV2
- 輸入格式：`1 x 28 x 28` 單通道灰階影像
- Batch size：64
- 最大訓練回合：100 epochs
- Loss：CrossEntropyLoss，label smoothing 為 0.1
- Optimizer：AdamW
- Learning-rate scheduler：OneCycleLR
- Early stopping patience：10
- Notebook 紀錄的最佳驗證準確率：約 91.57%

Notebook 會自動使用 CUDA；沒有可用 GPU 時則改用 CPU。建議使用 Google Colab
或具備 NVIDIA GPU 的環境執行。

## 類別順序

訓練資料使用以下英文類別名稱，順序不可任意更動：

```text
airplane, apple, banana, bicycle, bird, bus, camera, cat, chair, door,
eye, flashlight, guitar, hat, helmet, ice cream, jacket, mountain, star,
umbrella
```

英文名稱是 Quick, Draw! 資料集的下載 key。網站部署時，根目錄的
`labels.txt` 會依相同索引順序提供中文顯示名稱；翻譯 labels 不會改變模型輸出。

## 執行方式

1. 使用 Jupyter Notebook 或 Google Colab 開啟
   `114_2神經網路與深度學習_期末專案.ipynb`。
2. 依序執行所有儲存格。
3. Notebook 會下載各類別的 `.npy` 資料並建立訓練與驗證資料集。
4. 訓練完成後，確認最佳權重和類別順序正確。
5. 將部署權重放到專案根目錄，檔名使用 `quickdraw_20_classes.pt`。

## 部署注意事項

Notebook 目前仍將權重輸出為 `quickdraw_10_classes.pt`，這只是舊檔名，實際分類
頭為 20 類。部署前請將它重新命名為 `quickdraw_20_classes.pt`，並放到專案根目錄。

Notebook 產生的 `labels.txt` 是英文版本，不要直接覆蓋專案根目錄的中文
`labels.txt`。兩份 labels 必須保持完全相同的類別索引順序。

大型 `.npy` 資料、暫存輸出與中間 checkpoint 不建議提交到 Git。

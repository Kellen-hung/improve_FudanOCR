# 專案使用方法

本專案是 FudanOCR 論文「Chinese Text Recognition with A Pre-Trained CLIP-Like Model Through Image-IDS Aligning」的程式整理版本，主要分成兩個階段：

1. `CCR-CLIP/`：預訓練 CCR-CLIP 模型，學習中文字圖像與 IDS / radical 分解序列的對齊表示。
2. 根目錄程式：使用預訓練 CCR-CLIP 產生的中文字表示，訓練與執行中文文字辨識模型。

## 目錄說明

```text
.
├── CCR-CLIP/              # CCR-CLIP 預訓練程式
├── data/                  # 字典與 IDS / radical 分解檔案
├── docs/                  # 專案文件
├── model/                 # 文字辨識模型與 CLIP text encoder
├── config.py              # 根目錄文字辨識訓練 / 推論設定
├── main.py                # 文字辨識訓練入口
├── inference.py           # 文字辨識推論入口
├── train.py               # 文字辨識訓練流程
├── env.yaml               # Conda 環境設定
└── requirements.txt       # Python 套件版本參考
```

## 大檔案與 GitHub 放置原則

LMDB 訓練資料與模型權重檔案通常很大，不建議直接放進 GitHub。

建議只在 repo 中保留空目錄或說明檔，實際資料與權重由使用者自行下載或放到本機指定位置：

```text
lmdb/                 # 放 LMDB 資料集，不提交大型資料檔
pre_train_model/      # 放 CCR-CLIP 預訓練權重，不提交 .pth 大檔
```

若資料或權重路徑不同，請修改 `config.py` 與 `CCR-CLIP/config.py` 中的路徑。

## 安裝環境

建議使用 `env.yaml` 建立 Conda 環境：

```bash
conda env create -f env.yaml
conda activate CHR2
```

如果只使用 `pip`，可參考：

```bash
pip install -r requirements.txt
```

實際訓練與推論需要 PyTorch、CUDA、LMDB、OpenCV、Pillow、torchvision 等套件。請確認 CUDA 版本與本機 GPU 環境相容。

## 需要準備的檔案

### 1. 字典與分解檔案

根目錄文字辨識程式會讀取 `config.py` 中的下列檔案：

```python
'alpha_path' : './data/char_handwriting_Chinese.txt'
'radical_path': './data/radical_all_Chinese.txt'
'decompose_path': './data/decompose.txt'
```

`CCR-CLIP/` 預訓練程式會讀取 `CCR-CLIP/config.py` 中的：

```python
'alphabet_path': './data/radical_alphabet_27533_benchmark.txt'
'decompose_path': './data/decompose_27533_benchmark.txt'
```

執行 `CCR-CLIP/main.py` 時，工作目錄通常需要切到 `CCR-CLIP/`，讓 `./data/...` 能正確指向 `CCR-CLIP/data/...`。

### 2. LMDB 資料集

訓練資料使用 LMDB 格式，LMDB 中預期包含：

```text
num-samples
image-000000001
label-000000001
image-000000002
label-000000002
...
```

根目錄文字辨識訓練使用 `config.py`：

```python
'train_dataset' : '/path/to/train_lmdb'
'validation_dataset': '/path/to/val_lmdb'
```

`CCR-CLIP/` 預訓練使用 `CCR-CLIP/config.py`：

```python
'train_dataset': '/path/to/train_lmdb'
'test_dataset': '/path/to/val_lmdb'
```

如果有多個資料集，可以用逗號分隔：

```python
'train_dataset': '/path/to/train_a,/path/to/train_b'
```

### 3. 預訓練模型權重

根目錄文字辨識模型會使用 CCR-CLIP 預訓練權重：

```python
'pre-train_model': '/path/to/pre_train_model.pth'
```

如果要從既有文字辨識模型繼續訓練或推論，設定：

```python
'resume_model' : '/path/to/final_model.pth'
```

若從頭訓練文字辨識模型，可將 `resume_model` 設為空字串：

```python
'resume_model' : ''
```

## 預訓練 CCR-CLIP

先確認 `CCR-CLIP/config.py` 裡的資料集路徑、字典路徑與實驗名稱：

```python
config = {
    'epoch': 40,
    'train_dataset': '/path/to/train_lmdb',
    'test_dataset': '/path/to/val_lmdb',
    'batch': 128,
    'imageW': 128,
    'imageH': 128,
    'alphabet_path': './data/radical_alphabet_27533_benchmark.txt',
    'decompose_path': './data/decompose_27533_benchmark.txt',
    'max_len': 30,
    'lr': 1e-4,
    'exp_name': 'adult_with_notation',
}
```

執行：

```bash
cd CCR-CLIP
CUDA_VISIBLE_DEVICES=0 python main.py
```

訓練輸出會存到：

```text
CCR-CLIP/history/<exp_name>/
```

主要權重檔通常是：

```text
CCR-CLIP/history/<exp_name>/best_model.pth
```

完成後，可將該權重放到本機的 `pre_train_model/` 目錄，並在根目錄 `config.py` 中設定 `pre-train_model`。

## 訓練文字辨識模型

回到 repo 根目錄，修改 `config.py`：

```python
config = {
    'exp_name' : 'your_experiment_name',
    'epoch' : 100,
    'lr' : 1,
    'batch' : 128,
    'test' : False,
    'resume_model' : '',
    'train_dataset' : '/path/to/train_lmdb',
    'validation_dataset': '/path/to/val_lmdb',
    'pre-train_model': '/path/to/pre_train_model.pth',
    'alpha_path' : './data/char_handwriting_Chinese.txt',
    'radical_path': './data/radical_all_Chinese.txt',
    'decompose_path': './data/decompose.txt',
}
```

執行：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py
```

訓練輸出會存到：

```text
history/<exp_name>/
```

常見輸出：

```text
model.pth
best_model.pth
record.txt
result_file_validation_*.txt
```

注意：`main.py` 會呼叫 `saver()`，並重建 `history/<exp_name>/`。如果同名實驗資料夾已存在，內容會被清除後重新建立。

## 執行推論

先修改 `config.py`：

```python
'resume_model' : '/path/to/final_or_best_model.pth'
'pre-train_model': '/path/to/pre_train_model.pth'
'inference_dataset': './test1'
```

`inference_dataset` 應該是一個圖片資料夾，目前支援常見圖片副檔名：

```text
.jpg
.jpeg
.png
.bmp
```

執行：

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py
```

程式會逐張輸出：

```text
filename.png -> 辨識結果
```

## 常見設定項

### `config.py`

```python
'imageH' : 32
'imageW' : 256
```

文字辨識模型輸入圖片會 resize 到此大小。

```python
'char_len' : 60
```

推論時最多解碼字數。

```python
'batch' : 128
```

訓練與推論 batch size。若 GPU 記憶體不足，可調小。

### `CCR-CLIP/config.py`

```python
'imageH': 128
'imageW': 128
```

CCR-CLIP 預訓練輸入圖片大小。

```python
'max_len': 30
```

IDS / radical 序列最大長度。

## 建議 GitHub 結構

為了讓專案可以被 clone 後理解如何放資料，建議保留空目錄和 `.gitkeep` 或說明檔：

```text
lmdb/
pre_train_model/
```

但不要提交：

```text
*.pth
*.pt
*.ckpt
大型 LMDB 資料檔
```

可以在 README 或本文件中說明：

1. LMDB 資料請自行放到 `lmdb/` 或設定檔指定路徑。
2. CCR-CLIP 預訓練權重請自行放到 `pre_train_model/` 或設定檔指定路徑。
3. 訓練輸出的 `history/`、`runs/`、權重檔不建議提交到 GitHub。

## 最小使用流程

如果已經有 LMDB 資料和 CCR-CLIP 預訓練權重：

```bash
conda env create -f env.yaml
conda activate CHR2
```

修改 `config.py`：

```python
'train_dataset' : '/path/to/train_lmdb'
'validation_dataset': '/path/to/val_lmdb'
'pre-train_model': '/path/to/pre_train_model.pth'
'resume_model' : ''
```

訓練：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py
```

推論：

```python
'resume_model' : '/path/to/best_model.pth'
'inference_dataset': './test1'
```

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py
```

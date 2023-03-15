# 🎉Cathaysite Template Testing V0.0.1🎉

## 使用 Template 的目的:

1. 加速開發效率
2. 方便管理開發環境 🎉🎉🎉
3. 我就懶 😃😃😃

## Template 說明:

1. 先確認一下當前的 `python` 版本，若需要使用 `conda` 更新 `python` 版本，
   請使用 `conda create --name [conda env name] python=[python version] -c conda-forge`。
   Ex: 建立名為 **py3-11** 的環境且使用 **python 3.11** 版本: `conda create --name py3-11 python=3.11 -c conda-forge`。
   並使用 `conda activate [conda env name]`來啟動環境。

2. 請使用 `python -m venv venv` 來重新在資料夾內建立環境，完成後可執行 venv\Scripts 資料夾內的 **activate.bat** 來啟動環境。

3. 請使用 **requirements.txt** 來建立安裝基本 package，建議先看一下檔案內針對模型版本的部分，是否跟 GIT 的版本相同。(若是後續開發改用其他版本，也請先開 branch)

4. Happy modeling ~

---

## 以下是預設資料夾結構，參考[此網頁](https://dzone.com/articles/data-science-project-folder-structure)

- Data: 存放所有資料，依**模型版本**存放，個模型內有自己的子資料夾 train, predict and test
  - train: 存放訓練資料和生成訓練資料的 python 檔案，若訓練資料過大則可以不存
  - (未完待續)

```
📦project
│
└───📂data
│   └───📂model_YYYYMMDD
│       └───📂train
│       │   │   📜data.xlsx
│       │   │   📜require_cols.pickle
│       │   │   ...
│       │
│       └───📂predict
│       │   │   📜df_predict.feather
│       │   │   📜require_cols.pickle
│       │   │   ...
│       │
│       └───📂test
│           │   📜result.xlsx
│           │   ...
│
└───📂docs
│   │   📜document1.doc
│   │   📜document2.pdf
│   │   ...
│
└───📂models
│   └───📂model1
│       │   📜param_dict.pickle
│       │   📜model1.json
│       │   📜model1.json
│       │   ...
│
└───📂pipeline
│   │   📜data.py
│   │   📜train.py
│   │   📜test.py
│   │   ...
│
└───📂src
│   │   📜data_processing.py
│   │   📜outlook.py
│   │   📜toolbox.py
│
└───📂venv
│
│📜README.md
│📜main.py
│📜requirements.txt
│📜.gitignore
|   ...

```

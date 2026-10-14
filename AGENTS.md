# 專案組織與開發原則

本專案是 Robot Learning 的可執行知識庫，組織方式參考 `~/ai-lib/README.md`。修改前先閱讀本專案 `README.md`，遵循以下規則。

## 目錄責任

- `methods/`：可重用的方法與演算法實作。
- `models/`：值得獨立研究或重用的模型架構；僅供單一方法使用的小型網路可留在方法內。
- `paradigms/`：問題設定、假設、目標、概念與相關項目的連結。
- `datasets/`：資料來源、schema、切分、前處理與下載／載入程式。
- `benchmarks/`：任務、任務集合、評估協定、指標與分數解讀。
- `experiments/`：實際執行的實驗，負責環境／模擬器／機器人設定、訓練迴圈、超參數、checkpoint、繪圖與評估。
- `tests/`：跨項目的驗證；項目專屬測試可放在項目自己的 `tests/`。

## 實作規則

1. 直接使用頂層知識目錄，不增加 `src/robot_learning/` 包裝層。
2. 項目名稱使用有意義的 `snake_case`；依主要教學／實作目的選擇唯一主位置，其他關係以 README 連結表示。
3. 不要求統一的 `BaseEnv`、`BaseAgent`、`BaseAlgorithm`、`Trainer`、`Runner`、模型基底或生命週期 API。需要時使用框架原生抽象，例如 `torch.nn.Module`。
4. 每個項目依需求安排檔案；不預先建立未使用的程式、設定或抽象。先實作，看到實際重複再抽取共用邏輯。
5. 方法／模型放可重用邏輯，實驗擁有具體執行流程；不將單一環境的訓練流程硬塞進全專案框架。
6. 說明與程式放在一起；新增項目時撰寫 README。實驗應記錄依賴、執行命令、資料／模擬器要求、設定、seed 與評估方式。
7. 大型資料集、checkpoint、影片、紀錄及產生的輸出不加入 Git。`.gitignore` 規則可隨實際需求調整。
8. 頂層 `setup.py` 使用 namespace package 封裝；各實驗另行記錄自己的額外依賴。現有 AI-Lib 程式保留原路徑與 import，後續新增項目再依主要目的安排位置。

## 環境與驗證

- 環境由使用者自行處理；除非使用者另行要求，不建立／修改 conda 環境、不安裝套件、不啟動訓練。
- 搬入的程式依賴 PyTorch 與 Gymnasium；`train.sh` 保留來源的 conda 環境設定，不在搬移時自動執行。
- `gpu-base` 已於 2026-10-14 通過 GPU 矩陣運算、CPU 結果比對與反向傳播驗證。可重建的環境規格位於 `envs/gpu-base.yml`（Python 3.11 與固定 pip 套件版本；系統套件由 conda 依平台解析，GPU PyTorch 適用於支援的 Windows／Linux NVIDIA 環境）；重建／clone 命令見 README。基底未安裝 NumPy 與 Gymnasium，實驗需要時在獨立環境補上。不要修改基底環境；除非使用者要求，僅使用已有環境進行必要驗證。
- 執行與變更相符的驗證；搬移時可檢查來源檔案一致性、Python 語法與 import 路徑，不必啟動完整訓練。

# 專案命令執行方式

適用範圍：以下 PowerShell、MSYS2 zsh 路徑、環境設定與沙箱替代執行方式僅適用於這台電腦（主機名稱 DESKTOP-3O29EMG）。在其他電腦上，應先檢查當地 shell、conda、git 與沙箱是否正常，不要直接套用這些固定路徑或預設使用沙箱外執行。

## PowerShell、zsh、conda 與 git

此專案的 conda 與 git 已在 MSYS2 zsh 中驗證可用。不要假設它們存在於 PowerShell 的 PATH。

若 Codex 沙箱命令在啟動階段回報 `timed out ... connecting runner pipe-in`，這是執行器連線失敗，不能據此判定 zsh、conda 或 git 不存在。不要反覆重試相同的沙箱指令。透過工具的正式核准機制使用沙箱外 PowerShell（`sandbox_permissions: "require_escalated"`），並說明實際執行目的；這不代表沙箱已修復。

PowerShell 中的 `zsh` 目前解析為：

```text
C:\Users\user\AppData\Local\Microsoft\WindowsApps\zsh.cmd
```

這個啟動器會切到 `C:\Users\sheng-yu`，且不轉傳命令列參數。因此不要使用 `zsh -lc '...'` 或 `zsh -ic '...'` 執行自動化指令：即使沒有輸出且退出碼為 0，也不能視為指定命令已成功執行。

改從 PowerShell 直接呼叫啟動器指向的 zsh 執行檔，設定其所需環境，並在 zsh 中明確切回專案：

```powershell
$env:HOME = 'C:\Users\sheng-yu'
$env:TERM = 'xterm-256color'
Remove-Item Env:GIT_DIR,Env:GIT_WORK_TREE -ErrorAction SilentlyContinue
& 'C:\Users\sheng-yu\msys64\usr\bin\zsh.exe' -lic 'cd /d/sheng-yu/robot-learning && pwd && conda --version && git --version'
```

這裡的 HOME 是既有 zsh 啟動器所需的環境設定，不得用作暫存變數。清除 GIT_DIR 與 GIT_WORK_TREE 可避免繼承全域 Git 設定而指向錯誤儲存庫。實際作業時替換最後的檢查指令，保留明確的 `cd ... &&`，避免切換目錄失敗後仍繼續操作。

2026-10-14 已驗證：工作目錄 `/d/sheng-yu/robot-learning`、conda `26.7.1`（由 .zshrc 定義的 shell function）、git `/usr/bin/git`，版本 `2.55.0`。這些是當次觀察值，日後環境可能改變。

非互動工具執行 -lic 時可能出現 `can't change option: monitor` 和 `gitstatus failed to initialize`。當次 conda 與 git 仍正常執行；應檢查實際命令的輸出與退出碼，不要把提示元件初始化錯誤當成 Git 不可用，也不要為消除提示而擅自修改使用者的 .zshrc。

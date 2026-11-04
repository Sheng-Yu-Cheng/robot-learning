# 執行環境

本區保存實驗使用的環境定義與本機執行說明。環境應依實驗需求獨立管理；閱讀文件不需要建立環境或安裝套件。

- [gpu-base.yml](gpu-base.yml)：GPU 基底環境定義。
- [maniskill.yml](maniskill.yml)：ManiSkill 環境定義。

以下是特定電腦的命令執行設定；其他平台須先檢查當地工具，不照抄固定位置。這些設定不會自動建立或修改環境。

## 本機命令執行原則

- 以下設定僅適用於 DESKTOP-3O29EMG；在其他電腦先檢查當地 shell、conda、Git 與沙箱，不直接沿用固定路徑。
- conda 與 Git 透過 MSYS2 zsh 使用，不假設位於 PowerShell 的 PATH。
- PowerShell 的 zsh.cmd 啟動器不轉傳參數且會切換目錄；自動化不得使用它執行 zsh -lc 或 zsh -ic 命令。
- 直接呼叫 `C:\Users\sheng-yu\msys64\usr\bin\zsh.exe`，設定 HOME 為 `C:\Users\sheng-yu`、TERM 為 xterm-256color，清除繼承的 GIT_DIR 與 GIT_WORK_TREE。
- HOME 是啟動環境設定，不能當暫存變數；在 zsh 明確切回專案後，以成功條件串接後續操作，避免切換目錄失敗仍執行。
- 若沙箱在啟動時回報 timed out connecting runner pipe-in，視為執行器連線問題；不要反覆重試同一沙箱指令，也不要據此判定工具不存在。透過正式核准機制以沙箱外 PowerShell 執行並說明目的，不宣稱沙箱已修復。
- 非互動 zsh 的 monitor 或 gitstatus 初始化提示不等於命令失敗，依實際輸出及退出碼判斷；不為消除提示修改使用者的 shell 設定。

### PowerShell 呼叫範例

在上述電腦透過 MSYS2 zsh 檢查專案 Git 狀態：

```powershell
$env:HOME = 'C:\Users\sheng-yu'
$env:TERM = 'xterm-256color'
Remove-Item Env:GIT_DIR,Env:GIT_WORK_TREE -ErrorAction SilentlyContinue
& 'C:\Users\sheng-yu\msys64\usr\bin\zsh.exe' -c 'export PATH=/usr/bin:$PATH; cd /c/Users/sheng-yu/d/robot-learning && git status --short --ignore-submodules'
```

此範例只讀取 Git 狀態；實際命令的授權、輸入及產物位置仍依所屬任務與 README。不要使用不轉傳參數的啟動器代替 zsh 執行檔。

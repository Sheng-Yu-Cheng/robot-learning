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

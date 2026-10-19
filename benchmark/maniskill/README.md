# ManiSkill

整合 ManiSkill 操作任務、專家示範下載與回放工具。上游程式保留在 `maniskill/` Git submodule；本地 `replay.py` 將既有軌跡回放成可訓練的狀態觀測／動作資料及 MP4，供遠端查看。

以下指令從專案根目錄執行，使用已安裝本專案 editable package 與 ManiSkill 的環境：

```shell
conda activate maniskill
```

## 產出分類

此處的 `outputs/` 管理任務資料；訓練與策略評估結果放在所屬實驗的 `outputs/`，讓相同示範可供不同實驗共用。

```text
benchmark/maniskill/outputs/<task>/
├── motionplanning/                 下載的 motion planning 示範
├── rl/                             下載的 RL 示範、上游權重與範例影片
├── teleop/                         下載的遙操作示範
└── replays/<source>/<batch>/        回放後資料、metadata 與 videos/ 影片
```

原始下載保留上游的來源分類；replay 另按來源與批次存放，避免覆寫原始資料。新增任務建立新的 task 目錄，同任務的新 replay 建立新的 batch。HDF5 與配對 JSON 放在同一批次，回放影片集中於該批次的 `videos/`。

## 取得 ManiSkill 原始碼

首次 clone 主專案後，取回固定版本的上游 submodule：

```shell
git submodule update --init --recursive
```

環境需分別以 editable 模式安裝主專案與上游 ManiSkill；現有環境已安裝時不需重做：

```shell
python -m pip install --no-build-isolation -e .
python -m pip install --no-build-isolation -e ./benchmark/maniskill/maniskill
```

## 下載 demonstrations

列出可下載的任務：

```shell
python -m mani_skill.utils.download_demo
```

下載 PickCube 的專家示範：

```shell
python -m mani_skill.utils.download_demo PickCube-v1 --output_dir benchmark/maniskill/outputs
```

下載工具解壓任務資料到指定位置；目前 PickCube 的 motion planning 資料位於 `benchmark/maniskill/outputs/PickCube-v1/motionplanning/`，包含配對的 `trajectory.h5` 與 `trajectory.json`。已下載時直接使用現有檔案，不必重複下載。其他任務將 `PickCube-v1` 換成可下載的任務 ID，並依解壓後的實際來源位置指定 replay 輸入。

## Replay：匯出 50 筆可訓練示範與影片

使用現有 motion planning 軌跡，保存 50 筆成功回放的 demonstrations：

```shell
python -m benchmark.maniskill.replay --env-id PickCube-v1 --source motionplanning --traj-path benchmark/maniskill/outputs/PickCube-v1/motionplanning/trajectory.h5 --output-dir benchmark/maniskill/outputs/PickCube-v1/replays/motionplanning/demos_50_new --count 50
```

`--traj-path` 指向來源 HDF5，同目錄需保留 JSON；`--source` 記錄資料來源名稱。這是在模擬器重播已有動作，不是重新收集新的專家操作。程式保留來源 controller，以 CPU 模擬及 CPU 離屏渲染執行，補出狀態觀測，略過未成功的回放，直到存滿指定數量；若來源不足則回報錯誤。

輸出目錄包含：

- `trajectory.h5`、`trajectory.json`：配對的 observations、actions 與 metadata，可供訓練使用。
- `videos/000000.mp4` 等：每筆成功示範的影片。
- `manifest.json`：來源軌跡與輸出對應、回放進度。
- `verification.json`：完成後的資料對齊、影片影格數與時長檢查結果。

`--output-dir` 必須是尚未存在的新目錄。既有 `demos_50` 保留供訓練使用，上例另存為 `demos_50_new`；再次執行請換新批次名稱。新 task 也沿用 `outputs/<task>/replays/<source>/<batch>/` 分層。其他選項可查詢：

```shell
python -m benchmark.maniskill.replay --help
```

訓練與策略評估指令見 [Action Policy 實驗 README](../../experiments/maniskill_action_policy/README.md)。使用新 replay 批次時，將訓練的 `--demo-path` 指向該批次的 `trajectory.h5`。

## 隨機選取示範並補錄穩定段

`--shuffle-seed` 先隨機排列來源 episodes，再選成功回放；`--hold-steps` 在每筆來源動作結束後，於模擬器實際重複最後的絕對 joint position target，記錄後續觀測與動作。只有來源為 `pd_joint_pos` 時支援此選項，避免重複 delta command 造成持續位移。最後仍須通過環境成功判定才能保存。

```shell
python -m benchmark.maniskill.replay --traj-path benchmark/maniskill/outputs/PickCube-v1/motionplanning/trajectory.h5 --output-dir benchmark/maniskill/outputs/PickCube-v1/replays/motionplanning/demos_200_hold32 --count 200 --hold-steps 32 --shuffle-seed 42
```

這些延長的示範供模型學習穩定段；評估時不重播示範、不固定末端動作，也不根據抓取狀態切換策略。新增選項與來源設定記錄於 `manifest.json`。

### Joint delta 示範

可將來源的絕對 joint targets 逐步換算為原生 `pd_joint_delta_pos`，以目前模擬器的 joint positions 計算 delta，依 controller 範圍裁切後實際執行。轉換會改變 controller 與可能的軌跡，需重新檢查成功，不能只替換 metadata。這裡使用上游 controller 換算函式，不呼叫 IK。

```shell
python -m benchmark.maniskill.replay --traj-path benchmark/maniskill/outputs/PickCube-v1/motionplanning/trajectory.h5 --output-dir benchmark/maniskill/outputs/PickCube-v1/replays/motionplanning/demos_200_delta_hold32 --count 200 --hold-steps 32 --shuffle-seed 42 --control-mode pd_joint_delta_pos --video-count 3
```

`--hold-steps` 仍延續來源的最後絕對目標，每步重新換算 delta，不是重複同一 delta。`--video-count 3` 僅渲染並錄製前三筆成功示範，其餘只保存訓練資料，不呼叫渲染；省略則全部錄影。manifest 與 verification 中未錄影的 episode 記為 `video: null`。所有資料仍檢查成功、有限值與時間對齊，已錄製影片另檢查影格數與時長。

## CPU reset 的接觸狀態修正

同一 CPU scene 在成功抓取後直接 reset，可能在初始觀測讀到上一回合的接觸力；本機已重現普通 reset 的 `is_grasped=True`、原生重建 scene reset 的 `is_grasped=False`，同 seed 的其他 41 維觀測一致。這是 reset 邊界的資料問題，不代表機器人一開始真的抓住物件。

`replay.py` 現在設定 ManiSkill 原生 `reconfiguration_freq=1`，每回合重建 scene，並確認 reset 回傳 `reconfigure=True`。程式不把抓取標記硬改成 false，也不新增任何 policy 行為。manifest 記錄 `reset_strategy` 與各 episode 的 `reset_reconfigured`。重建 scene 會增加 reset 成本。

既有 `demos_50`、`demos_200_hold32`、`demos_200_delta_hold32` 尚未重建，初始觀測仍分別有 49/50、199/200、199/200 筆異常；第一步動作後該欄位均恢復 false。舊的 verification 只檢查對齊、有限值、成功與影片，不代表已檢查初始接觸語意。需要重播到新批次，不能只修改 manifest 宣稱已修復，例如：

```shell
python -m benchmark.maniskill.replay --traj-path benchmark/maniskill/outputs/PickCube-v1/motionplanning/trajectory.h5 --output-dir benchmark/maniskill/outputs/PickCube-v1/replays/motionplanning/demos_200_delta_hold32_reset_fixed --count 200 --hold-steps 32 --shuffle-seed 42 --control-mode pd_joint_delta_pos --video-count 3
```

已依上述命令重建新批次：200 筆的初始抓取標記皆為 false，全部最後 32 步保持成功；來源順序與前版一致。詳細檢查見該批次的 reset_audit.json。修正前的資料與 checkpoint 保留原樣，重訓與評估見實驗 README。

## v3：局部擾動與恢復監督

新增的 `--perturbation-std` 在 joint delta controller 的正規化 command 空間注入稀疏 Gaussian 擾動。每一步的監督標籤仍以原始專家 absolute joint target 與當下實際受擾 qpos 換算，實際行為則執行帶噪 command；最後明確保留的 hold 段不加噪音。這是參考噪音注入示範的局部恢復收集，不是完整 DART 的自適應噪音最佳化，也不是能從任意失敗狀態救回物件的 oracle。[DART 原始研究](https://proceedings.mlr.press/v78/laskey17a.html)

```shell
python -m benchmark.maniskill.replay --traj-path benchmark/maniskill/outputs/PickCube-v1/motionplanning/trajectory.h5 --source-manifest benchmark/maniskill/outputs/PickCube-v1/replays/motionplanning/demos_200_delta_hold32_reset_fixed/manifest.json --output-dir benchmark/maniskill/outputs/PickCube-v1/replays/motionplanning/recovery_180_delta_v3 --count 180 --hold-steps 32 --control-mode pd_joint_delta_pos --video-count 3 --perturbation-std 0.1 --perturbation-probability 0.15 --perturbation-seed 123
```

`--source-manifest` 固定來源群組與順序；此例只從前版 200 筆來源中收集，未引入額外 task seed。每筆 HDF5 的 I/O 語意：

| 欄位 | 意義 |
| --- | --- |
| `obs` | 真實受擾模擬狀態，形狀 `[T+1,42]` |
| `actions` | 對應當下狀態的專家修正標籤，`[T,8]`，供訓練使用 |
| `executed_actions` | 實際執行的帶噪行為，`[T,8]`，對應已記錄的狀態轉移與影片 |
| `success` | 真實模擬器在行為動作執行後的成功判定 |

metadata 明確標記 `action_semantics="expert_labels"`，不能把 `actions` 當作原本的真實執行動作重播；本地 replay 會拒絕直接將這種資料當來源。重播真實行為需使用 `executed_actions`，不能只改欄位名稱或 metadata 宣稱軌跡等價。原始示範與一般無擾動 replay 的 `actions` 語意保持不變。

本批次成功收集 180 筆，包含 2,133 次擾動，初始抓取標記全部正確，全部最後 32 步保持成功。`recovery_audit.json` 記錄標籤與受擾 qpos 的數值對齊、保持段與來源分組檢查。只保留成功恢復的軌跡會有篩選偏差，這批資料並不涵蓋所有掉落或嚴重偏離情況。

## TCP delta position 示範

`replay.py --control-mode pd_ee_delta_pos` 將原始 absolute joint reference 做 forward kinematics，取得期望 TCP 位置，減去真實當下 TCP 在 robot root frame 的位置，再依原生 controller 的位置上下限正規化。輸出 action 為 `[delta_x, delta_y, delta_z, gripper]`，共四維；Panda 的 translation 每軸 `[-1,1]` 對應 `[-0.1,0.1]` 公尺。此模式保留當下姿態，沒有旋轉輸出。原生 IK 決定關節目標，沒有自訂策略或測試時專家介入。

這會改變控制器與物理軌跡，不能將舊 joint action 直接截成前三維或改 metadata。超出原生位置範圍的參考目標最多執行四個實際控制步，每步重新計算 delta 並記錄狀態；保持段持續追蹤最後的原始 TCP 位置目標。只保存最後成功的回合；不保證所有 joint 專家軌跡都能用無旋轉的 controller 重現。`manifest.json` 記錄 TCP frame、尺度、動作維度與轉換規則，`verification.json` 驗證實際資料及影片。

CPU TCP controller 額外需要 pinocchio；本機 Windows 使用 conda-forge 套件，並在 `maniskill` 環境測試原生 IK。執行方式：

```shell
conda activate maniskill
python -m benchmark.maniskill.replay --traj-path benchmark/maniskill/outputs/PickCube-v1/motionplanning/trajectory.h5 --output-dir benchmark/maniskill/outputs/PickCube-v1/replays/motionplanning/demos_200_tcp_delta_hold32_v4 --count 200 --hold-steps 32 --shuffle-seed 42 --control-mode pd_ee_delta_pos --video-count 3
```

若比較同一組來源，可另外傳入 `--source-manifest` 指向前版 manifest；若該來源中成功軌跡不足，程式回報未達數量，不自動擴增來源。可搭配既有擾動參數收集 TCP 空間的局部恢復監督；`actions` 與 `executed_actions` 的區分與前述 recovery 規則一致。TCP 與 joint 資料不可混用，對應的訓練／評估及四維 I/O 架構見 [實驗 README](../../experiments/maniskill_action_policy/README.md)。
Windows 的 TCP IK 依賴以 `conda-forge::pinocchio=3.4` 管理，搭配 OpenBLAS，避免新引入的 MKL／LLVM OpenMP 與 PyTorch runtime 同時初始化；不設定 `KMP_DUPLICATE_LIB_OK`。可重建設定已記錄於 `envs/maniskill.yml`。既有環境的補裝命令為：

```shell
conda install -n maniskill -c conda-forge pinocchio=3.4 "libblas=*=*openblas" "libopenblas=*=*pthreads*"
```
TCP 正式批次已完成 200 筆、22,022 個動作，來源 ID 與順序和前版 reset-fixed joint 批次完全相同；本批次沒有需要額外控制子步的目標。初始 `is_grasped` 為 0/200，全部最後 32 步成功。以記錄的 TCP observation 獨立重算每一步標籤，最大正規化誤差約 1.79e-6；詳細結果保存於該批次 `tcp_audit.json`。三段正式示範影片皆已驗證非空、影格數及時長合理：

- [TCP 示範影片](outputs/PickCube-v1/replays/motionplanning/demos_200_tcp_delta_hold32_v4/videos/000000.mp4)
- [正式資料稽核](outputs/PickCube-v1/replays/motionplanning/demos_200_tcp_delta_hold32_v4/tcp_audit.json)
## TCP recovery 的正式對照批次

`recovery_180_tcp_delta_v5` 沿用 TCP primary 的來源 manifest，以正規化 command std 0.1、機率 0.15、seed 123 注入局部行為擾動，保留原生專家修正標籤，供 loss／恢復資料分別控制的實驗使用：

```shell
conda activate maniskill
python -m benchmark.maniskill.replay --traj-path benchmark/maniskill/outputs/PickCube-v1/motionplanning/trajectory.h5 --source-manifest benchmark/maniskill/outputs/PickCube-v1/replays/motionplanning/demos_200_tcp_delta_hold32_v4/manifest.json --output-dir benchmark/maniskill/outputs/PickCube-v1/replays/motionplanning/recovery_180_tcp_delta_v5 --count 180 --hold-steps 32 --control-mode pd_ee_delta_pos --video-count 3 --perturbation-std 0.1 --perturbation-probability 0.15 --perturbation-seed 123
```

這批資料的 `actions` 為四維教師標籤、`executed_actions` 為四維受擾行為；最後明確保持段停止加噪。沿用來源群組防止 train／validation 洩漏，不把帶噪行為當作教師，也不混入舊八維 joint 資料。完成檢查保存於批次的 `verification.json` 與 `recovery_audit.json`；影片集中於 `videos/`。本批次是局部專家擾動，未收集模型自身嚴重失敗後的修正，不能視為全域恢復能力。訓練、共同選權重及測試指令見 [實驗 README 的 v5 對照](../../experiments/maniskill_action_policy/README.md)。
本批次已完成 180 筆／19,829 個動作、2,144 次擾動，沒有跳過來源回放；全部最後成功。176 筆最後 32 步全成功，其餘四筆最後連續成功 31 步。初始抓取標記為 0/180；以記錄的受擾 TCP 位置獨立重算教師標籤，最大正規化誤差約 1.79e-6。最後保持段的行為與教師動作一致，已停止加噪。來源分組的 train／validation 軌跡數為 340／40，來源交集為零。稽核詳見該批次 `recovery_audit.json`，不將「最後成功」誤寫成每筆完整保持段都成功。
# Action Policies

這裡放置根據觀測歷史生成一段未來動作的策略模型。

- **Diffusion Policy**：學習去除動作中的噪音，透過逐步去噪產生動作序列。
- **Flow Matching Policy**：學習從噪音流向示範動作的速度場，以積分方式產生動作序列。

兩種策略使用條件式時間網路，提供訓練目標與動作採樣。目前接收狀態或已編碼的觀測特徵，資料處理、環境互動與完整訓練流程由實驗負責。

## 模型與學習選項

`diffusion_policy.py` 與 `flow_matching_policy.py` 提供 `compute_loss()`、`sample()` 與 `predict_action()`；`common.py` 負責輸入檢查與 mask loss。`networks.py` 提供兩種可選架構：

- `architecture="unet"`：原本的時間卷積 U-Net，保留既有 checkpoint 相容性。
- `architecture="mlp"`：獨立的觀測編碼器、時間編碼與三層 residual MLP，聯合生成整段未來動作；`hidden_dim` 控制寬度。

Flow Matching 額外支援 `temporal_loss_weight`：以預估乾淨動作的相鄰差分對齊示範差分，不將所有動作強制壓成常數。推論仍由模型生成動作，沒有任務階段判斷、手寫抓取動作或輸出平滑器。

模型由 [ManiSkill Action Policy 實驗](../../../experiments/maniskill_action_policy/README.md) 的訓練與評估入口使用，完整執行指令見該文件。

Flow Matching 的 `prediction_type="sample", solver="euler"` 直接學習乾淨端點 x1，以 `(x1_pred - x_t) / (1-t)` 定義速度場；訓練採相當於 `(1-t)^2` 加權 velocity matching 的端點 MSE。Euler 採樣只計算 t<1，最後一步直接到達模型預測的端點，不在 t=1 做除法。預設 `prediction_type="velocity"` 保留原本速度場目標與 Heun／Euler 選項。兩種模式皆可加入示範差分目標，不能直接比較不同目標的 loss。

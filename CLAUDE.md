# MiroFish

社交媒體預測模擬平台 — 透過 LLM 驅動的 Agent 模擬社群行為，產生預測報告。

## 工作準則

- **禁止敷衍或 workaround**：遇到問題時必須追查 root cause，不可以用「這是預期行為」「這只是 noise」等說法搪塞。如果當下無法確認，應該先驗證再回答，而非猜測一個看似合理的答案。
- **驗證優先於推測**：對任何異常現象（error log、非預期結果、edge case），先用程式碼或指令實際驗證，確認事實後再下結論。
- **質疑答案，不斷省思**：這就是最好的答案? 我已經拿出全力了? 還能夠更好嗎?

## 技術棧

- **Backend**: Python / Flask（`backend/`）
- **Frontend**: Vue 3 + Vite（`frontend/`）
- **LLM**: OpenAI SDK 格式（支援 OpenAI、阿里百煉、Ollama）
- **知識圖譜**: NetworkX + SQLite 混合架構
- **模擬引擎**: OASIS（Twitter / Reddit 平台模擬）

## 開發環境

- Python venv：`backend/.venv/`
- 環境變數：根目錄 `.env`（參考 `.env.example`）
- 本地 LLM：Ollama `qwen3.5:9b`

## LLM 客戶端 (`backend/app/utils/llm_client.py`)

`LLMClient` 提供 `chat()` 和 `chat_json()` 兩個方法。

**Ollama 0.17+ 本地端點限制**：

- **禁止 `extra_body={"think": bool}`**：Ollama 0.17 將 `false` 誤讀為開啟 thinking，導致 content 為空
- **禁止 `max_tokens`**：thinking + content 合計算，thinking 會吃光額度
- **禁止 `response_format`**：干擾 thinking 模型輸出
- Thinking 控制改用 `/no_think` prompt 指令（注入 system + user message）

**Thinking 模式**（`LLM_THINK` 環境變數，預設 `false`）：
- 結構化輸出 / Agent 行動 → `think=False`
- 分析彙整 / 報告生成 → `think=True`

## 資料存儲

| 資料 | 存儲位置 |
|------|---------|
| 知識圖譜 | SQLite `backend/uploads/graph_store.db`（啟動時載入至 in-memory NetworkX） |
| 模擬狀態 | `backend/uploads/simulations/sim_xxx/state.json` |
| 專案/文件 | `backend/uploads/projects/proj_xxx/` |
| 報告 | `backend/uploads/reports/report_xxx/` |

## 主要服務（`backend/app/services/`）

| 服務 | 用途 |
|------|------|
| `networkx_graph_store` | 知識圖譜存儲（啟動時自動從 SQLite 恢復） |
| `graph_builder` | LLM 實體抽取 + 圖譜構建 |
| `ontology_generator` | 本體論生成 |
| `simulation_config_generator` | 模擬配置生成 |
| `oasis_profile_generator` | Agent 角色生成 |
| `report_agent` | 報告生成與對話（需 `think=True`） |

## 慣例

- Commit 訊息使用英文，格式遵循 conventional commits（`feat:`, `fix:`, `docs:` 等）

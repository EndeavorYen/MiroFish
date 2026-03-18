# MiroFish

社交媒體預測模擬平台 — 透過 LLM 驅動的 Agent 模擬社群行為，產生預測報告。

## 技術棧

- **Backend**: Python / Flask，位於 `backend/`
- **Frontend**: Vue 3 + Vite，位於 `frontend/`
- **LLM**: 統一使用 OpenAI SDK 格式（支援 OpenAI、阿里百煉、Ollama 等）
- **記憶圖譜**: Zep
- **模擬引擎**: OASIS（支援 Twitter / Reddit 平台模擬）

## 開發環境

- Python venv 位於 `backend/.venv/`
- 環境變數配置在根目錄 `.env`（參考 `.env.example`）
- 本地 LLM 使用 Ollama，模型為 `qwen3.5:9b`

## 關鍵架構

### LLM 客戶端 (`backend/app/utils/llm_client.py`)

`LLMClient` 是所有 LLM 呼叫的統一入口，提供 `chat()` 和 `chat_json()` 兩個方法。

**Thinking 模式切換**（適用於 Ollama 本地模型如 qwen3.5）：

- 環境變數 `LLM_THINK=true|false` 控制全局預設（預設 `false`）
- 每次呼叫可透過 `think` 參數覆蓋：`None`=用預設、`True`=強制開、`False`=強制關
- 透過 OpenAI SDK 的 `extra_body={"think": bool}` 傳遞給 Ollama
- 僅對本地端點生效（`is_local` 判斷），雲端 API 不受影響
- 回應中的 `<think>` 標籤會被正則清除，作為安全網

使用原則：
- 結構化輸出 / Agent 行動 → `think=False`（省 token、快速回應）
- 分析彙整 / 報告生成 → `think=True`（深度推理）

### 配置 (`backend/app/config.py`)

- `Config.is_local_openai_compatible_base_url()` 偵測本地 Ollama 端點
- 本地端點自動使用較長 timeout（600s vs 180s）及佔位 API key（`"ollama"`）

### 主要服務

| 服務 | 路徑 | 用途 |
|------|------|------|
| `simulation_config_generator` | `backend/app/services/` | 生成模擬配置（JSON） |
| `oasis_profile_generator` | `backend/app/services/` | 生成 Agent 角色（JSON） |
| `report_agent` | `backend/app/services/` | 報告生成與對話（需推理） |
| `zep_tools` | `backend/app/services/` | Zep 記憶圖譜工具 |
| `ontology_generator` | `backend/app/services/` | 本體論生成 |

## 慣例

- Commit 訊息使用英文，格式遵循 conventional commits（`feat:`, `fix:`, `docs:` 等）
- LLM 回應會做 `<think>` 標籤清理和 markdown 代碼塊清理（處理不同模型的輸出差異）
- 截斷偵測：`finish_reason == 'length'` 時有修復邏輯
- 重試時 temperature 遞減（提高穩定性）

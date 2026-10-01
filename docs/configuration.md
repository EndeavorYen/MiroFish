# 設定（環境變數）

後端從環境變數與 `.env` 讀設定：`backend/app/config.py` 在啟動時載入專案根目錄的 `.env`，而且 **`.env` 的值會蓋過 shell 裡 export 的同名變數**（`load_dotenv(override=True)`）；要臨時改設定，請改 `.env`。多數人只需要設一個 profile：

```env
MIROFISH_PROFILE=local        # 全本機、零 decode 的路徑
# MIROFISH_PROFILE=local-hybrid  # 用本機 LLM 準備，模擬同 local
# MIROFISH_PROFILE=local-llm     # 本機 LLM 決定 agent 行動
```

profile 只會填「還沒設定」的模式設定，所以 `.env` 裡明確寫的值優先；服務位址例外：本機 profile 會把雲端的 `LLM_BASE_URL`／`EMBED_BASE_URL` 換成本機服務，`SYSTEM_ONE_BASE_URL` 若不是本機位址也會改成 `LLM_BASE_URL`，並移除 `LLM_BOOST_*`（見 `backend/app/profiles.py`）。

`backend/tests/test_env_documented.py` 會掃描程式讀取的每個環境變數，沒有寫在這份文件裡的測試就會失敗。

## Profile

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `MIROFISH_PROFILE` | 不設 | `local`、`local-hybrid`、`local-llm`；不設就不套用 profile，未知名稱會報錯。 |

## 模型服務

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `LLM_API_KEY` | 無 | LLM 的 API key；本機服務可以填任意值（profile 填 `local`）。 |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI 相容的 LLM 端點；本機 profile 為 `http://127.0.0.1:8000/v1`。 |
| `LLM_MODEL_NAME` | `gpt-4o-mini` | LLM 模型名稱；本機 profile 為 `qwen3.5-4b`。 |
| `LLM_BOOST_API_KEY` | 無 | 雲端路徑：Reddit agent 用的第二個 LLM 端點（本機 profile 會移除）。 |
| `LLM_BOOST_BASE_URL` | 無 | 同上，端點。 |
| `LLM_BOOST_MODEL_NAME` | 無 | 同上，模型名稱。 |
| `OPENAI_API_KEY` | 由 `LLM_API_KEY` 設定 | 模擬程序在啟動時依 `LLM_*` 設定給 CAMEL／OASIS 使用，不需要自己設。 |
| `OPENAI_API_BASE_URL` | 由 `LLM_BASE_URL` 設定 | 同上。 |
| `SYSTEM_ONE_BACKEND` | `local` | System One 讀出：`local`＝在本機模型服務讀 logprobs；`http`＝POST `{base}/v1/systemone`。 |
| `SYSTEM_ONE_BASE_URL` | `http://localhost:8000/v1` | System One 讀出用的模型服務；使用本機 profile 時，非本機位址會被改成 `LLM_BASE_URL`。 |
| `SYSTEM_ONE_MODEL` | `LLM_MODEL_NAME`，否則 `qwen3.5-4b` | System One 讀出用的模型名稱。 |
| `SYSTEM_ONE_API_KEY` | 無 | System One 端點的 API key。 |
| `SYSTEM_ONE_TOP_K` | `20` | 讀出時取的 logprobs 數量。 |
| `SYSTEM_ONE_PROMPT_FORMAT` | `chatml` | 讀出提示的格式：`chatml` 或 `plain`。 |
| `SYSTEM_ONE_ENSEMBLE` | 不設 | 讀出集成（#54）：`all` 或以逗號分隔的問題鍵；需搭配 `MODEL_POOL`。 |
| `MODEL_POOL` | 不設 | 多模型池（#48），JSON；不設就只用 `LLM_*` 一個模型。格式見 `docs/local-first.md`。 |

## 圖譜與向量

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `GRAPH_BACKEND` | `zep` | `zep`（Zep Cloud）或 `local`（本機 SQLite 圖譜）；本機 profile 為 `local`。 |
| `GRAPH_DATA_DIR` | `backend/uploads/graphs` | 本機圖譜的資料夾。 |
| `GRAPH_EXTRACTOR` | `local` | 本機圖譜的抽取器：`local`（零 decode）或 `stub`（測試用）。 |
| `LOCAL_NER` | `candidates` | 候選名稱來源：`candidates`（規則，零 decode）、`gliner`、`decode`（每段加一次小模型列名）。 |
| `LOCAL_NER_GLINER_MODEL` | `urchade/gliner_multi-v2.1` | `LOCAL_NER=gliner` 用的模型。 |
| `EXTRACT_SUMMARY_LLM` | `0` | `1`＝實體摘要改用小模型生成（預設用模板）。 |
| `GRAPH_EMBEDDER` | `http` | `http`（OpenAI 相容的 embeddings 端點）或 `hash`（無模型，測試與離線基準用）。 |
| `EMBED_BASE_URL` | `http://localhost:8001/v1` | embedding 服務；本機 profile 為 `http://127.0.0.1:8001/v1`。 |
| `EMBED_MODEL_NAME` | `intfloat/multilingual-e5-small` | embedding 模型名稱。本機 llama.cpp 要用正確斷詞的 GGUF（見 `docs/local-first.md`）。 |
| `EMBED_API_KEY` | 無 | embedding 端點的 API key。 |
| `EMBED_QUERY_PREFIX` | 依模型 | 查詢前綴；不設時 e5 模型用 `query: `。 |
| `EMBED_PASSAGE_PREFIX` | 依模型 | 文件前綴；不設時 e5 模型用 `passage: `。 |
| `ZEP_API_KEY` | 無 | `GRAPH_BACKEND=zep` 時必填。 |
| `ZEP_API_URL` | 不支援 | 設了會被拒絕：MiroFish 只連 Zep Cloud。 |

## 準備（本體、角色、模擬設定）

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `ONTOLOGY_MODE` | `llm` | `llm` 或 `template`（System One 選模板，不 decode）。 |
| `PROFILE_MODE` | `llm` | `llm` 或 `structured`（System One 讀出角色設定）。 |
| `SIM_CONFIG_MODE` | `llm` | `llm` 或 `structured`。 |
| `PREP_HOURS_PRIORS` | `0` | 活躍時段的先驗：`0` 不用、`1` 用內建檔、其他值是 `fit_hours_priors.py` 產生的檔案路徑。 |
| `STANCE_CALIBRATION` | 內建檔 | 立場校準檔的路徑；檔案不存在就不校準。 |

## 模擬

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `SIM_DECISION_BACKEND` | `llm` | agent 行動由誰決定：`llm`（CAMEL agent）或 `system_one`（零 decode 讀出）。 |
| `SIM_DECISION_CONCURRENCY` | `4` | System One 決策同時送出的請求數。 |
| `SIM_AGENT_CONTEXT_TOKENS` | 本機 `3072`；雲端不限 | LLM agent 記憶的 token 上限；`0`／`off` 關閉。 |
| `SIM_ACTION_PRIORS` | 內建檔 | 行動先驗：`off`、`default` 或檔案路徑（`fit_action_priors.py`）。 |
| `SIM_STANCE_PRIOR_WEIGHT` | `0.5` | 立場＝權重 × 角色設定的立場 ＋ (1 − 權重) × 當回合的讀出；0–1。 |
| `SIM_EMOTION_ALPHA` | `0.7` | 情緒更新的平滑係數。 |
| `SIM_FAIL_ROUNDS` | `2` | 同一平台連續幾個回合的決策全部失敗就中止模擬（#62）。 |
| `SIM_FAIL_RATIO` | `0.2` | 累計決策失敗比例超過多少就中止（#62）。 |
| `SIM_FAIL_MIN_DECISIONS` | `20` | 累計多少次決策後才開始看失敗比例（#62）。 |
| `SIM_RECSYS` | `oasis`；本機 profile `light` | 推薦系統：`light`（不載入 torch、twhin-bert）或 `oasis`（OASIS 原版）（#65）。 |
| `OASIS_DEFAULT_MAX_ROUNDS` | `10` | 未指定時的最大回合數。 |

## 貼文內容（System One 路徑）

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `CONTENT_MODE` | `tiered` | `tiered`（分層：模板、共享生成、少量完整生成）或 `template`（只用模板，不 decode）。 |
| `CONTENT_DECODE_BUDGET_PER_ROUND` | `120` | 每個平台每回合可用於生成貼文的 decode token 預算（剛好一則完整生成或一次共享生成）。 |
| `CONTENT_TOP_K_PERCENT` | `10` | 每回合最重要的前百分之幾的貼文可以完整生成。 |
| `CONTENT_BANK_SHARE` | `0` | 模板貼文改用立場句庫句子的比例；預設句庫只用來引導共享與完整生成的語氣。 |
| `CONTENT_STANCE_BANK` | `1` | `0`＝不用立場句庫。 |
| `CONTENT_STANCE_CHECK` | `1` | `0`＝不檢查生成貼文的立場。 |
| `CONTENT_STANCE_DITHER` | `0` | `1`＝在立場上加小幅抖動。 |
| `CONTENT_LANG` | 依模擬需求 | `zh` 或 `en`；其他值時依模擬需求的文字判斷。 |
| `CONTENT_SCRIPT` | `auto` | 中文模板依模擬需求用繁體或簡體；`off` 保留模板原本的簡體。 |

## 報告

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `REPORT_MODE` | `agent` | `agent`（ReportAgent）或 `metrics`（確定性的指標報告＋短摘要）；本機 profile 為 `metrics`。 |
| `REPORT_AGENT_MAX_TOOL_CALLS` | `5` | ReportAgent 每節最多的工具呼叫。 |
| `REPORT_AGENT_MAX_REFLECTION_ROUNDS` | `2` | ReportAgent 的反思回合數。 |
| `REPORT_AGENT_TEMPERATURE` | `0.5` | ReportAgent 的溫度。 |

## 後端服務

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `FLASK_HOST` | `0.0.0.0` | 後端監聽位址。 |
| `FLASK_PORT` | `5001` | 後端埠號。 |
| `FLASK_DEBUG` | `False` | `true`（不分大小寫）開啟 debug；`1` 不會開啟。不要用在正式環境。 |
| `RUNS_DB_PATH` | `backend/uploads/runs/runs.sqlite` | runs API（#63）與背景任務（圖譜建構、準備、報告，#68）共用的 SQLite 資料庫：run 的狀態、產物 id、事件與任務進度。後端重啟後仍查得到；重啟時還在進行的任務標為失敗（`error: backend restarted`），完成超過一天的任務在啟動時清除。 |
| `RUNS_SIM_MEMORY_MB` | `500` | runs API 多個 seed 平行時，每個模擬進程預估占用的主機記憶體（MB）。實測本機路徑約 300 MB。 |
| `RUNS_MEMORY_RESERVE_MB` | `2048` | 平行跑 seed 時保留給模型服務、作業系統與瀏覽器的主機記憶體（MB）。同時跑的 seed 數 = (可用記憶體 − 保留) ÷ 每個模擬，至少 1。 |
| `SECRET_KEY` | `mirofish-secret-key` | Flask secret key；對外服務時請改掉。 |
| `WERKZEUG_RUN_MAIN` | 由 Flask 設定 | debug reloader 的子程序標記，不需要自己設。 |
| `PYTHONIOENCODING` | 由後端設定 | 模擬子程序的輸出編碼（`utf-8`），不需要自己設。 |
| `PYTHONUTF8` | 由後端設定 | 同上，讓子程序的 `open()` 預設 UTF-8。 |

## 評估與研究腳本

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `MIROFISH_METRICS_DIR` | 依呼叫者 | LLM 使用量紀錄（`llm_usage.jsonl`）的資料夾。 |
| `AB_MIN_FREE_GB` | `2.5` | 批次評估時，可用記憶體低於這個值就不啟動下一個模擬。 |
| `HF_HOME` | `~/.cache/huggingface` | `convert_e5_gguf.py` 找 Hugging Face cache 的位置。 |
| `HF_HUB_CACHE` | `$HF_HOME/hub` | 同上，直接指定 hub cache。 |

## Docker compose 與前端

這些變數不是後端讀的。compose 的變數寫在專案根目錄的 `.env`；前端的變數由 Vite 讀取，本機開發要寫在 `frontend/.env`（或 shell 環境），Docker 映像則經由 `env_file: .env` 帶入：

| 變數 | 預設 | 說明 |
| --- | --- | --- |
| `LOCAL_LLM_MODEL` | `SubSir/Qwen3.5-4B-AWQ` | `docker compose --profile local` 的 vLLM 模型（16 GB 可改 `Qwen/Qwen3.5-4B`）。 |
| `LOCAL_LLM_GPU_MEMORY_UTILIZATION` | `0.90` | vLLM 預先占用的 VRAM 比例。 |
| `LOCAL_EMBED_MODEL` | `intfloat/multilingual-e5-small` | compose 的 TEI embedding 模型。 |
| `HF_TOKEN` | 無 | 下載需要授權的 Hugging Face 模型時使用。 |
| `VITE_API_BASE_URL` | 開發：同源（Vite proxy）；建置：`http://localhost:5001` | 前端呼叫後端 API 的位址；開發與建置時都會讀取。寫在根目錄 `.env` 對 `npm run dev` 無效。 |

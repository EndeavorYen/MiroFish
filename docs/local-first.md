# 本機優先（Local-First）模式

MiroFish 原本依賴雲端 LLM 與 Zep Cloud。本機模式讓整條流程都在一張 16GB（本機路徑只需約 7GB）的消費級 GPU 上執行，不呼叫任何外部 API：知識圖譜、準備、模擬、報告都在本機完成。

本文件的所有數據都在 RTX 5080 16GB、Qwen3.5-4B（Q4_K_M，llama.cpp）、multilingual-e5-small（CPU）上實測；情境與指令見文末。

## 兩種本機設定

| 設定 | 做什麼 | 適合 |
| --- | --- | --- |
| `MIROFISH_PROFILE=local` | 全本機；agent 決策用 System One（讀 logprobs，不 decode）、分層內容、模板與結構化準備、metrics 報告 | 快、便宜、VRAM 小；大量情境掃描 |
| `MIROFISH_PROFILE=local-llm` | 全本機；agent 決策與準備仍用本機 LLM decode（記憶有預算，8K/slot 可跑） | 行為最接近原本的 LLM 路徑 |

兩者都只會補上「沒設定」的變數：`.env` 或環境中明確設定的值永遠優先。沒有設定 `MIROFISH_PROFILE` 時行為完全不變（預設仍是雲端 LLM + Zep）。

## 快速開始（llama.cpp，已實測）

1. 下載 [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) 與模型：`Qwen3.5-4B-Q4_K_M.gguf`、`multilingual-e5-small-F16.gguf`。
2. 啟動模型服務（兩個終端）：

   ```bash
   # 生成與 System One 讀出（GPU）
   llama-server -m Qwen3.5-4B-Q4_K_M.gguf -c 65536 -np 8 -ngl 99 --jinja \
     --reasoning off --no-mmproj --alias qwen3.5-4b --host 127.0.0.1 --port 8000

   # 向量（CPU）
   llama-server -m multilingual-e5-small-F16.gguf --embedding --pooling mean -ngl 0 \
     -c 2048 -b 2048 -ub 2048 -np 4 --alias intfloat/multilingual-e5-small \
     --host 127.0.0.1 --port 8001
   ```

3. 在 `.env` 加上一行（其餘本機預設由 profile 補上）：

   ```env
   MIROFISH_PROFILE=local        # 或 local-llm
   ```

4. 照原本方式啟動：`npm run dev`。

### llama-server 參數注意事項

- `-c 65536 -np 8` 是每個 slot 8K context。本機路徑與 `local-llm`（有記憶預算）都在這個設定下實測零錯誤。
- **不要加 `--kv-unified`**：共用 KV pool 在多個 agent 同時請求時會回 `Context size has been exceeded`，丟失回合。
- ReportAgent（`REPORT_MODE=agent`）單一請求會超過 8K；本機建議用 `REPORT_MODE=metrics`（兩個 profile 都已預設）。
- 若要跑沒有記憶預算的 LLM 決策（`SIM_AGENT_CONTEXT_TOKENS=off`），每個 slot 需要 64K（`-c 262144 -np 4`，約 13GB VRAM）。

### Docker compose（未在本機驗證）

`docker compose --profile local up` 會啟動 vLLM（`local-llm`，`--max-model-len 8192`）與 TEI（`local-embed`）。vLLM 以 Hugging Face 模型 ID 作為模型名稱，所以要在 `.env` 另外設定：

```env
LLM_MODEL_NAME=SubSir/Qwen3.5-4B-AWQ
SYSTEM_ONE_MODEL=SubSir/Qwen3.5-4B-AWQ
```

## 各元件與開關

| 元件 | 變數 | 本機值 | 說明 |
| --- | --- | --- | --- |
| 知識圖譜 | `GRAPH_BACKEND` | `local` | SQLite + FTS5（CJK bigram）+ sqlite-vec，RRF 混合檢索；不需要 `ZEP_API_KEY` |
| 實體抽取 | `GRAPH_EXTRACTOR` / `LOCAL_NER` | `local` / `candidates` | 規則候選 + System One，零 decode；`decode` 會多一次小模型列名（未看過資料的召回 0.61 → 0.87） |
| 本體 | `ONTOLOGY_MODE` | `template` | 模板 + System One 選擇，零 decode |
| 人設 | `PROFILE_MODE` | `structured` | System One 結構化欄位 + 模板文字，零 decode |
| 模擬設定 | `SIM_CONFIG_MODE` | `structured` | System One 讀出活躍度與時段，零 decode |
| agent 決策 | `SIM_DECISION_BACKEND` | `system_one` | 階層式 System One 讀出 + action priors，零 decode |
| 貼文內容 | `CONTENT_MODE` | `tiered` | 模板／共享生成／完整生成三層，每平台每回合 decode 預算 `CONTENT_DECODE_BUDGET_PER_ROUND`（預設 120） |
| action priors | `SIM_ACTION_PRIORS` | `default` | 校正小模型「偏向發文」的讀出；`off` 關閉 |
| 報告 | `REPORT_MODE` | `metrics` | 確定性指標報告 + 一段 ≤800 tokens 摘要 |
| LLM 決策記憶 | `SIM_AGENT_CONTEXT_TOKENS` | `3072`（本機 URL 時的預設） | 只在 `local-llm` 有作用 |

## 實測結果（golden scenario，每組 5 個 seed、24 回合）

### 成本

| 項目 | LLM 路徑（本機模型） | 本機路徑（`local`） |
| --- | ---: | ---: |
| 準備階段 decode（本體＋人設＋設定） | 約 19,000 | **0** |
| 模擬每回合 decode | 462 | **31**（6.7%） |
| 報告 decode | 4,238（ReportAgent） | **~200**（metrics） |
| VRAM（llama-server 8K/slot） | 7.2 GB | 7.2 GB |
| 有動作回合平均延遲 | 11.0 s | 13.3 s |

### 忠實度（閘門 G4，B = 本機路徑、A = LLM 路徑）

| 條件 | 結果 | 門檻 |
| --- | --- | --- |
| 每回合 decode B／A | 0.067 ✅ | ≤ 0.10 |
| 動作分布 JS | 0.043 ✅ | ≤ 2 × A 組間（0.054） |
| 各角色平均立場相關 | 0.524 ✅ | ≥ 0.5 |
| 立場分布 JS | 0.047 ❌ | ≤ 2 × A 組間（0.006） |
| VRAM | 7.2 GB ✅ | ≤ 10 GB |
| 每 run agent 動作數 | 132.6 對 164.6（81%） | — |
| distinct-2 | 0.350 對 0.360 | — |

結論：本機路徑在 **動作行為、各角色立場、成本** 上達標；**貼文的立場分布** 與 LLM 路徑仍有差異（模板與共享內容的立場表達較集中），因此建議：

- **預設維持 LLM 路徑**；
- 需要大量、快速、低成本模擬（情境掃描、參數敏感度分析）時用 `MIROFISH_PROFILE=local`；
- 需要最接近原行為、又不想用雲端時用 `MIROFISH_PROFILE=local-llm`。

### 第二情境：泛化檢查（東濱一號離岸風場，9 個角色，每組 5 個 seed）

action priors 只在 golden 上擬合，這裡沒有針對新情境做任何調整。

| 指標 | golden | 離岸風場 |
| --- | --- | --- |
| 準備階段 decode（本機路徑） | 0 | 0 |
| decode B／A | 0.067 ✅ | 0.091 ✅ |
| 動作分布 JS（門檻：2 × A 組間） | 0.043 ≤ 0.054 ✅ | 0.067 ≤ 0.100 ✅ |
| 每 run 動作數 B／A | 81% | 80% |
| 各角色平均立場相關（A 組間） | 0.524（0.955）✅ | 0.445（0.455）❌ |
| 立場分布 JS（A 組間） | 0.047（0.003）❌ | 0.044（0.010）❌ |

動作行為與成本的結果在新情境上成立。小情境的各角色立場指標，連 A 自己都只有 0.455 的一致性，B 已經達到這個水準；兩個情境一致未通過的是貼文立場分布。

## 已知限制

- 立場分布閘門未通過：本機路徑的貼文立場比 LLM 路徑集中（內容生成層的問題，決策層已校正）。
- action priors 在 golden 的 calibration seeds 上擬合，已在第二情境驗證動作分布，但情境仍只有兩個。
- 本機路徑每 run 的 agent 動作數約為 LLM 路徑的 80%（LLM agent 每次啟用平均做 1.6–2 個動作）。
- 線上執行不是逐位元可重播（OASIS 共用全域亂數、模型伺服器批次不確定）；可重播的是活躍 agent 的選擇與 System One 的取樣。

## 重現實驗

```bash
cd backend
# 準備（LLM 與模板各一次；--fixture 可換情境）
uv run python scripts/golden_pipeline.py prepare --work <A> --prep-mode llm
uv run python scripts/golden_pipeline.py prepare --work <B> --prep-mode template
# A/B（兩組各 5 個 seed）
uv run python scripts/ab_eval.py --work-a <A> --work-b <B> --out <out> --seeds 1 2 3 4 5
# action priors 重新擬合（calibration seeds，勿用評估 seeds）
uv run python scripts/fit_action_priors.py --a-runs <A runs> --b-runs <B runs> \
  --out app/simulation_policy/action_priors.json
```

相關 issue：#1（epic）、#2–#13、#19、#26–#28、#34–#36。

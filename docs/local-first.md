# 本機優先（Local-First）模式

MiroFish 原本依賴雲端 LLM 與 Zep Cloud。本機模式讓整條流程都在一張 16GB（本機路徑只需約 7GB）的消費級 GPU 上執行，不呼叫任何外部 API：知識圖譜、準備、模擬、報告都在本機完成。

本文件的所有數據都在 RTX 5080 16GB、Qwen3.5-4B（Q4_K_M，llama.cpp）、multilingual-e5-small（CPU）上實測；情境與指令見文末。

## 兩種本機設定

| 設定 | 做什麼 | 適合 |
| --- | --- | --- |
| `MIROFISH_PROFILE=local` | 全本機；agent 決策用 System One（讀 logprobs，不 decode）、分層內容、模板與結構化準備、metrics 報告 | 快、便宜、VRAM 小；大量情境掃描 |
| `MIROFISH_PROFILE=local-llm` | 全本機；agent 決策與準備仍用本機 LLM decode（記憶有預算，8K/slot 可跑） | 行為最接近原本的 LLM 路徑 |

模式類設定（決策、內容、準備、報告）只在沒設定時補上，`.env` 仍可覆寫。服務端點不同：本機 profile 保證不呼叫外部 API，所以 `.env` 裡若是雲端的 `LLM_BASE_URL`／`EMBED_BASE_URL`（例如 `.env.example` 附的 DashScope），會被換成本機服務並記錄警告，`LLM_BOOST_*` 會被移除；本機的其他 port 或模型名稱則保留，System One 跟著使用。沒有設定 `MIROFISH_PROFILE` 時行為完全不變（預設仍是雲端 LLM + Zep）。

## 用本機路徑掃描，再用 local-llm 確認

`local` 路徑的成本約是 LLM 路徑的 5–10%。它能答對的是方向類結論，不是逐則貼文的細節。評估庫（6 個情境，每組 5 個 seed，閘門 G5）上，它和 LLM 路徑的一致程度如下：

| 結論 | 和 LLM 路徑一致的情境 | 報告裡的可信度 |
| --- | --- | --- |
| 主要陣營（多數角色偏支持、中立或反對） | 6/6 | 高 |
| 整體傾向（全部貼文平均立場） | 3/6 | 中 |
| 走向（最後一段減第一段） | 3/6 | 低 |
| 角色排序（誰最支持、誰最反對） | 2/6 | 低；角色立場幾乎相同時為「無法區分」 |

使用方式：

1. **單次模擬**：`MIROFISH_PROFILE=local` 的 metrics 報告多了「掃描結論與可信度」一節。每項結論都附上可信度。準備階段給各角色的立場幾乎相同時（標準差 < 0.10），角色排序會直接標成「無法區分」，不列名單。這個門檻是在 calibration 情境選定的；評估庫上它標出的 3 個情境（港灣、青浦、Northbridge），角色排序也確實都低於 0.5。
2. **一次掃多個情境、多個 seed**：

   ```bash
   cd backend
   uv run python scripts/scan.py --work <準備好的目錄> [<目錄> ...] --out <輸出> --seeds 1 2 3
   ```

   每個情境會列出主要陣營有幾個 seed 一致、整體傾向與走向的平均和標準差，以及角色排序在兩半 seed 之間的穩定度。遇到主要陣營不一致、走向正負不一致、角色無法區分或排序不穩時，建議欄會寫明要用 `local-llm` 確認。
3. **確認**：挑出來的情境用 `MIROFISH_PROFILE=local-llm` 重跑。

seed 之間一致，只代表本機路徑自己穩定，不代表它和 LLM 路徑一致。golden 的走向在 5 個 seed 都是 −0.50 左右，LLM 路徑卻幾乎持平；澄川的角色排序在 seed 之間很穩（0.84），和 LLM 路徑卻只有 0.39。所以要依走向或角色排序下判斷時，一律用 `local-llm` 確認。

## 方案比較：發布前先比較幾個方案的反應（#56）

實際要做的決定，通常不是「會有幾 % 的人反對」，而是「A、B、C 哪個方案反彈最小」。例如公告措辭、定價或漲價方案、政策草案版本、危機回應稿。比較方案時，所有方案用同一組 seed 配對，貼文都用同一題評分，模型本身的偏誤在方案之間大多會抵消。所以相對比較比絕對數字可靠。

```bash
cd backend
# options.json：[{"name": "baseline", "text": "..."}, {"name": "free_trial", "text": "..."}]，第一個是基準
uv run python scripts/compare_options.py --fixture <情境目錄> --options options.json --out <輸出> --seeds 1 2 3
# 用 LLM 準備與 LLM 決策確認排名
uv run python scripts/compare_options.py ... --path llm
```

每個方案都會變成一個情境：基準的種子文件加上方案內容，需求也寫明方案。每個方案各自經過零 decode 的準備，所以方案會影響各角色的立場，再跑同一組 seed。報告列出每個方案的整體傾向、反對貼文比例、和基準的配對差距（有幾個 seed 同向，全部同向才算可區分），以及最反對的貼文。

**敏感度測試**（golden，空中計程車試點，每個方案 3 個 seed、24 回合，本機路徑）：

| 方案 | 內容 | 整體傾向 | 反對貼文 | 對基準 |
| --- | --- | ---: | ---: | --- |
| baseline | 按原計畫實施 | 0.68 | 24% | — |
| favorable | 前三個月免費、晚上十點後停飛、避開住宅區、先辦說明會、補助司機轉職 | 0.76 | 7% | 反對 −17.0 個百分點（3/3 seed 同向） |
| unfavorable | 票價為計程車三倍、24 小時營運、飛越住宅區、不徵詢居民、無補助 | 0.51 | 44% | 傾向 −0.17、反對 +19.6 個百分點（3/3 seed 同向） |

方案確實會傳進準備階段：有司機補助時，計程車工會的準備立場從 0.08 升到 0.38；航線避開住宅區時，住宅區業主聯誼會只從 0.10 升到 0.15。最反對的貼文也跟著方案變：favorable 最強的反對只是「安全與法規要先評估」（0.24），unfavorable 則是「絕對不能容許、立即叫停」（0.01）。

限制：
- 這是模擬裡的反應，不是對真實世界的預測。
- 做決定前，前兩名請用 `--path llm` 確認。本機與 LLM 路徑的排名一致率見下方驗證。
- seed 少時，只有全部同向才算數：3 個 seed 全部同向，在沒有差異時也有 1/4 的機率發生，差距小時請用 5 個 seed。

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

3. 在 `.env` 加上一行（其餘本機預設由 profile 補上；`.env` 裡原本的雲端 LLM 設定會被換成本機，不必手動刪除）：

   ```env
   MIROFISH_PROFILE=local        # 或 local-llm
   ```

4. 照原本方式啟動：`npm run dev`。

### llama-server 參數注意事項

- `-c 65536 -np 8` 是每個 slot 8K context。本機路徑與 `local-llm`（有記憶預算）都在這個設定下實測零錯誤。
- **不要加 `--kv-unified`**：共用 KV pool 在多個 agent 同時請求時會回 `Context size has been exceeded`，丟失回合。
- ReportAgent（`REPORT_MODE=agent`）單一請求會超過 8K；本機建議用 `REPORT_MODE=metrics`（兩個 profile 都已預設）。
- 若要跑沒有記憶預算的 LLM 決策（`SIM_AGENT_CONTEXT_TOKENS=off`），每個 slot 需要 64K（`-c 262144 -np 4`，實測閒置 VRAM 13.0 GB）。

### Docker compose（已實測，2026-09-29）

`docker compose --profile local up local-llm local-embed` 會啟動 vLLM（`--max-model-len 8192`）與 TEI，使用 port 8000／8001。後端請在主機上跑（`npm run dev`）。`mirofish` 容器裡的 `127.0.0.1` 指的是容器自己，所以要改用 `LLM_BASE_URL=http://host.docker.internal:8000/v1` 這類位址。vLLM 以 Hugging Face 模型 ID 當作模型名稱，要在 `.env` 設定（System One 會沿用）：

```env
LLM_MODEL_NAME=SubSir/Qwen3.5-4B-AWQ
```

實測環境：Docker Desktop 28.4、vLLM v0.30.0、TEI cpu-1.8.1、RTX 5080 16GB、主機 RAM 32 GB，`.wslconfig` 設 `memory=10GB`。

- **端到端通過**：同一支 Playwright 腳本、golden 種子、10 回合，耗時 1.4 分鐘，報告頁 6/6 章節。
- System One 可以直接讀 vLLM 的 logprobs。`system_one_eval.py`（205 題）的結果：choice 0.946、noul 0.877、score 1.00，標籤覆蓋率 ≥ 0.99，G2 通過。TEI 的 `/v1/embeddings` 回傳 384 維。
- 主機記憶體：沒有限制時，WSL VM 會長到 15 GB，主機只剩 2.8 GB，後端被結束。限制成 10 GB 後，vLLM 用 3.4 GB、TEI 用 1.5 GB。TEI 在預設設定下（每核心一個 tokenization worker、暖機 batch 很大）會在暖機時被 OOM 結束，compose 已改成 `--max-batch-tokens 2048 --tokenization-workers 4`。
- VRAM：vLLM 預設 `--gpu-memory-utilization 0.90`，會預先占掉整張卡（實測連同其他程式共 15.4 GB）。要符合 10 GB 預算，請設 `LOCAL_LLM_GPU_MEMORY_UTILIZATION` 約 0.6，這個設定還沒實測。
- 前端在開發模式（Docker 映像也是）改走 Vite 的 `/api` proxy。直接連 `:5001` 時，瀏覽器會重用開發伺服器已關閉的 keep-alive 連線，`POST /api/simulation/prepare` 因此偶爾出現 `ERR_CONNECTION_RESET`，環境準備那一步就會一直等下去。

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
| 貼文立場檢查 | `CONTENT_STANCE_CHECK` | `1` | 生成後用零 decode 讀出實際立場；與意圖不同級時，改用最多 3 句同級模板中最接近的一句（#45） |
| 語氣範例庫 | `CONTENT_STANCE_BANK` / `CONTENT_BANK_SHARE` | `1` / `0` | `locales/<lang>_stance_bank.json`（只取自 calibration 情境的 LLM runs）；預設只當共享、完整生成 prompt 的同級語氣參考（#45、#52） |
| 準備階段校正 | `STANCE_CALIBRATION` | 未附檔（恆等） | `fit_prep_calibration.py` 可擬合立場保序映射與活躍度位移；兩者在評估情境都沒有整體改善，所以預設不附校正檔（#46、#47，見已知限制） |
| 多模型池 | `MODEL_POOL` / `SYSTEM_ONE_ENSEMBLE` | 未設定（單一模型） | 見下方「多模型池」（#48） |
| 時段先驗 | `PREP_HOURS_PRIORS` | `0`（關閉） | `1` 用 `app/services/hours_priors.json`（`fit_hours_priors.py`，只用 calibration 情境擬合）。抽樣時走向改善、角色排序變差，所以預設關閉（#53） |
| 立場級別抽樣 | `CONTENT_STANCE_DITHER` | `0`（關閉） | `1`：依意圖在相鄰兩級之間隨機抽，期望值等於意圖。抽樣結果有好有壞，所以預設關閉（#53） |

## 實測結果（golden scenario，每組 5 個 seed、24 回合）

### 成本

| 項目 | LLM 路徑（本機模型） | 本機路徑（`local`） |
| --- | ---: | ---: |
| 準備階段 decode（本體＋人設＋設定） | 約 19,000 | **0** |
| 模擬每回合 decode | 462 | **30**（6.5%） |
| 報告 decode | 4,238（ReportAgent） | **~200**（metrics） |
| VRAM（llama-server 8K/slot） | 7.2 GB | 7.2 GB |
| 有動作回合平均延遲 | 11.0 s | 12.9 s |

### 忠實度（閘門 G4 v2，B = 本機路徑、A = LLM 路徑）

現行閘門（#44）用 seed 層級 bootstrap（預設 1,000 次，RNG seed 0）的 95% 信賴區間，不再用固定的 0.5 與「2 × 平均雜訊」：

- 動作 JS、立場分布 JS：B vs A 的 bootstrap 分布對 A 組間分布做單尾檢定，p < 0.05 判失敗。
- 各角色立場：共同角色少於 12 為「無法判讀」（不算失敗）；否則 B vs A 相關的信賴區間下界要 ≥ A 組間相關的中位數 − 0.15，且共同角色 ≥ A 的 80%。
- 情境庫彙總：≥ 80% 的情境通過才建議切換，≥ 50% 為有條件切換。單一情境的「通過」含「無法判讀」。

下表是改成 v2 之前、用固定門檻記下的 golden 實測（門檻欄是當時的規則）。

| 條件 | 結果 | 門檻 |
| --- | --- | --- |
| 每回合 decode B／A | 0.065 ✅ | ≤ 0.10 |
| 動作分布 JS | 0.028 ✅ | ≤ 0.054（= 2 × A 組間 0.027） |
| 各角色平均立場相關 | 0.809 ✅（A 組間 0.953） | ≥ 0.5 |
| 立場分布 JS | 0.048 ❌ | ≤ 0.01（2 × A 組間 0.003，下限 0.01） |
| VRAM | 7.2 GB ✅ | ≤ 10 GB |
| 每 run agent 動作數 | 138.4 對 164.6（84%） | — |
| distinct-2 | 0.374 對 0.360 | — |

結論：本機路徑在 **動作行為、各角色立場、成本** 上達標；**貼文的立場分布** 與 LLM 路徑仍有差異（強烈反對的貼文偏少：golden 1.9% 對 12.7%），因此建議：

- **預設維持 LLM 路徑**；
- 需要大量、快速、低成本模擬（情境掃描、參數敏感度分析）時用 `MIROFISH_PROFILE=local`；
- 需要最接近原行為、又不想用雲端時用 `MIROFISH_PROFILE=local-llm`。

### 第二情境：泛化檢查（東濱一號離岸風場，9 個角色，每組 5 個 seed）

action priors 只在 golden 上擬合，這裡沒有針對新情境做任何調整。

| 指標 | golden | 離岸風場 |
| --- | --- | --- |
| 準備階段 decode（本機路徑） | 0 | 0 |
| decode B／A | 0.065 ✅ | 0.079 ✅ |
| 動作分布 JS（門檻：2 × A 組間） | 0.028 ≤ 0.054 ✅ | 0.065 ≤ 0.100 ✅ |
| 每 run 動作數 B／A | 84% | 73% |
| 各角色平均立場相關（A 組間） | 0.809（0.953）✅ | 0.144（0.514）❌ |
| 立場分布 JS（A 組間） | 0.048（0.003）❌ | 0.032（0.010）❌ |

動作行為與成本的結果在新情境上成立。離岸風場只有 9 個角色、多數立場中立，各角色立場相關在同一份程式碼的兩次評估間從 0.527 擺到 0.144（A 自己也只有 0.51），這個指標在小情境上無法判讀；兩個情境一致未通過的是貼文立場分布。

### Phase 2 起點（G4 v2，6 個情境，每組 5 個 seed、24 回合）

2026-09-27，RTX 5080、llama-server `-c 65536 -np 8`（8K/slot，沒有 `--kv-unified`）。總耗時 15825 秒（4.4 小時）。六個情境的 `runs_with_llm_errors` 都是空的。通過 0/6，建議 `keep_llm`。

| 情境 | 結果 | decode B／A | 動作 JS（p） | 各角色立場（下界，門檻，共同角色） | 立場分布 JS（p） | VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| golden_scenario | 未通過 | 0.061 ✅ | 0.035（0.042）❌ | 0.815（0.601 < 0.736，14）❌ | 0.042（0.000）❌ | 9227 ✅ |
| qingpu_clinic | 未通過 | 0.074 ✅ | 0.059（0.000）❌ | 0.224（−0.019 < 0.623，26）❌ | 0.033（0.000）❌ | 9295 ✅ |
| gangwan_curriculum | 未通過 | 0.104 ❌ | 0.061（0.000）❌ | 0.577（0.355 < 0.435，17）❌ | 0.012（0.447）✅ | 8581 ✅ |
| chengchuan_power | 未通過 | 0.078 ✅ | 0.054（0.000）❌ | 0.443（0.356 < 0.724，30）❌ | 0.086（0.000）❌ | 8761 ✅ |
| northbridge_measles | 未通過 | 0.043 ✅ | 0.222（0.000）❌ | 0.039（−0.434 < 0.352，16）❌ | 0.131（0.000）❌ | 7938 ✅ |
| fengqiao_recall | 未通過 | 0.062 ✅ | 0.091（0.001）❌ | 0.516（0.159 < 0.472，17）❌ | 0.051（0.000）❌ | 7938 ✅ |

成本閘門大多通過（港灣課程的 decode B／A 是 0.104，剛過 0.10）。動作 JS 與各角色立場在六個情境都未過 v2：golden 的立場點估計仍是 0.815，但 95% 信賴區間下界 0.601 低於 A 組間中位數 − 0.15（0.736）。沒有情境是「無法判讀」（共同角色都 ≥ 12）。

### Phase 2 收尾（G4 v2，同一套情境與 A 組 runs，B 用 #43 收尾分支重跑）

2026-09-28，同樣的硬體與 server 設定。A 組沿用上表的 LLM runs，B 組每組 5 個 seed、24 回合，重新準備並重跑。每個情境都用自己的 `stance_question.txt` 評分。`runs_with_llm_errors` 都是空的。通過 0/6，建議 `keep_llm`。

| 情境 | decode B／A | 動作 JS（p） | 各角色立場（下界，門檻） | 立場分布 JS（p） | 每 run 動作數 B／A | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| golden_scenario | 0.046 ✅ | 0.037（0.035）❌ | **0.903（0.809 ≥ 0.737）✅** | 0.030（0.000）❌ | 0.73 | 8032 ✅ |
| qingpu_clinic | 0.066 ✅ | 0.033（0.004）❌ | 0.026（−0.262 < 0.633）❌ | 0.063（0.000）❌ | 0.90 | 8602 ✅ |
| gangwan_curriculum | **0.092 ✅** | 0.050（0.001）❌ | 0.280（0.112 < 0.417）❌ | 0.008（0.856）✅ | 0.96 | 7849 ✅ |
| chengchuan_power | 0.070 ✅ | 0.028（0.019）❌ | 0.517（0.459 < 0.725）❌ | 0.066（0.000）❌ | 1.16 | 7849 ✅ |
| northbridge_measles | 0.069 ✅ | 0.185（0.000）❌ | 0.169（−0.110 < 0.324）❌ | 0.218（0.000）❌ | 0.62 | 7849 ✅ |
| fengqiao_recall | 0.056 ✅ | 0.073（0.001）❌ | 0.734（0.505 < 0.508）❌ | 0.072（0.000）❌ | 0.70 | 7849 ✅ |

和起點相比：
- 動作 JS 在 6 個情境都下降，但只有 golden 的 p 值接近 0.05。
- 各角色立場在 4 個情境上升，golden 通過；楓橋的下界只差門檻 0.003。
- 立場分布 JS 在 golden、港灣、澄川下降，在青浦、Northbridge、楓橋上升。
- decode B／A 全部 ≤ 0.10。

G4 v2 要求 B 和 A 的差距落在 A 組 seed 之間的雜訊內，而 A 的雜訊很小（立場分布組間 JS 0.004–0.008）。B 的差距減小之後，仍然判定失敗。產品方向因此改成「便宜、方向夠準的掃描工具」，重要情境用 `local-llm` 確認。掃描工具用的閘門另開 issue 定義。

### 掃描工具閘門 G5（2026-09-28 寫定，在計算任何 G5 數字之前）

本機路徑的角色改成「便宜、方向夠準的掃描工具」。G5 只檢查掃描使用者會拿來下判斷的方向類結論，並且用絕對容差，不再要求 B 和 A 的差距落在 A 的 seed 雜訊之內。所有值都取 5 個 seed 的平均。

| 閘門 | 量什麼 | 通過條件 |
| --- | --- | --- |
| 角色排序 | B、A 各角色平均立場的 Spearman 相關 | ≥ 0.5；共同角色 < 12 時標為「無法判讀」 |
| 主要陣營 | 各角色依平均立場分成反對（< 0.4）、中立、支持（> 0.6），比較兩邊人數最多的陣營 | B 與 A 相同 |
| 整體傾向 | 全部貼文的平均立場（0..1） | 兩邊相差 ≤ 0.10 |
| 走向 | 每 4 回合平均立場，最後一段減第一段 | 兩邊的變化量相差 ≤ 0.10 |
| 成本 | 沿用 G4：decode B／A、VRAM | ≤ 0.10、≤ 10 GB |

動作 JS 與立場分布 JS 只報告（標出 ≤ 0.05），不列入判定。所有閘門通過或「無法判讀」，該情境就算通過；≥ 80% 的情境通過才算「可以當掃描工具」。計算用 `scripts/scan_gates.py`，直接讀 `ab_report.json`，不需要重跑模擬。走向只比兩組都有貼文的 4 回合區段：B 的第一則貼文可能比 A 早一段，從不同起點相減會高估變化。

G5 結果（和上面兩張表是同一批 runs）：

| 閘門 | #44 起點 | Phase 2 收尾 |
| --- | --- | --- |
| 角色排序 ≥ 0.5 | 2/6（golden 0.86、港灣 0.64） | 2/6（golden 0.86、楓橋 0.54） |
| 主要陣營相同 | 5/6（Northbridge ❌） | **6/6** |
| 整體傾向相差 ≤ 0.10 | 4/6 | 3/6 |
| 走向相差 ≤ 0.10 | 3/6 | 3/6 |
| 成本 | 5/6（港灣 decode 0.104 ❌） | **6/6** |
| 情境通過 | 0/6 | 0/6，判定 not_ready |

主要陣營六個情境都和 LLM 路徑一致，成本也都在預算內。還不能當掃描工具的原因是角色排序：青浦 0.20、Northbridge 0.15、澄川 0.39、港灣 0.45。走向也有落差，golden 的 B 在最後一段轉向反對（−0.37），A 則幾乎不變（+0.03）。

### #53：角色排序與走向的原因分析與嘗試

**角色排序**。LLM 路徑的排序本身很穩（A 組 seed 兩半互比 0.66–0.91），所以 B 的差距不是雜訊造成的。拆開來看有三種情況：

- **準備標籤分不開角色**：青浦的準備立場標準差 0.06，Northbridge 0.003（情境本身就是「多數中立」）。B 的貼文跟著準備標籤走，標籤沒有差異，排序就是雜訊（B 組 seed 兩半互比只有 0.09 和 0.26）。Northbridge 的 A 雖然有排序，但 A 自己的準備也全部是 0.50。它的排序來自那一次 LLM 準備寫出的人設文字，本機路徑沒有合理的方式重現。
- **小模型讀錯陣營**：澄川的摘要明明寫著「站在漁民這邊」（反對方），準備讀出卻是 0.84。
- **從意圖到貼文的流失**：calibration 情境上，意圖對 A 的排序中位數是 0.50，寫成貼文後只剩 0.35。

試過、但都沒有預設啟用的做法：

| 做法 | 選擇依據 | 結果 |
| --- | --- | --- |
| 準備讀出的變體：去掉角色條件、陣營題、Phi、Qwen+Phi 集成 | calibration 情境上，準備讀出和 A 貼文立場的排序相關 | 都沒有勝過現行讀法（中位數 0.47） |
| 依角色類型把立場向中立收縮 | calibration 擬合 | 評估庫上沒有整體改善，青浦變差 |
| `CONTENT_STANCE_DITHER`：依意圖隨機抽相鄰的級別，讓級別的期望值等於意圖 | 抽樣：seeds 1–3、24 回合，6 個情境 | 3 個情境變好、3 個變差（港灣 0.40→0.16） |
| `PREP_HOURS_PRIORS`：時段讀出依 calibration 擬合的權重重新加權 | 同上 | 走向 2/6→3/6，但角色排序有 4 個情境變差 |

**走向**。晚上 20–23 點對應最後幾回合。B 的時段讀出把 63% 的實體放在上班時段（LLM 準備是 48%），晚間覆蓋只有 A 的一半左右。最後一段因此只剩少數、而且偏向一方的角色在發文。時段先驗能補回一部分（golden 的走向 −0.27→−0.19），但不夠，而且會讓角色排序變差。

結論：G5 維持 0/6（收尾那一次的完整基準），預設行為沒有改變。報告改成在每個結論旁標出可信度，並指出哪些要用 `local-llm` 確認（見上方「用本機路徑掃描，再用 local-llm 確認」）。

## 已知限制

- 立場分布閘門仍未通過（見上表）。golden 的強烈反對從 3% 升到 6%（A 10%）。一面倒支持的情境（青浦、楓橋）則變得更中立或更強烈支持，和 A 的差距變大。
- Northbridge（英文）的本機貼文仍有 96% 是中立（A 52%）。準備階段的立場與每則貼文的立場讀出幾乎都是中立。英文模板、英文意圖題與語氣範例庫都沒能讓意圖本身帶有立場（#52）。
- 準備階段的保序立場映射與活躍度位移都擬合過（`fit_prep_calibration.py`），但在評估情境沒有整體改善，所以預設不附校正檔（#46、#47）。活躍度位移能補回動作偏少的情境（golden、楓橋），但會把已經對齊的情境推過頭。
- 每 run 的動作數 B／A 在各情境差很多（0.62–1.16）。每次啟用的動作數已經和 A 對齊（#47），差距來自每回合的啟用次數，而這在不同情境間的差異無法用單一參數修正。
- B 的 run 間差異很大：每 run 約 60–90 則貼文，golden 同一版程式的強烈反對可從 1% 到 12%。3 個 seed 的抽樣曾經誤導判斷，決定前要用每組 5 個 seed 的完整基準。
- 線上執行不是逐位元可重播（OASIS 共用全域亂數、模型伺服器批次不確定）；可重播的是活躍 agent 的選擇與 System One 的取樣。

## 重現實驗

```bash
cd backend
# 準備（LLM 與模板各一次；--fixture 可換情境）
uv run python scripts/golden_pipeline.py prepare --work <A> --prep-mode llm
uv run python scripts/golden_pipeline.py prepare --work <B> --prep-mode template
# 單一情境 A/B（兩組各 5 個 seed）
uv run python scripts/ab_eval.py --work-a <A> --work-b <B> --out <out> --seeds 1 2 3 4 5
# 情境庫整套（tests/fixtures/scenarios/suite.json；可續跑）
uv run python scripts/ab_suite.py --out <suite-out>
# 開發迭代：每組 3 個 seed、12 回合
uv run python scripts/ab_suite.py --out <suite-out> --quick
# action priors 重新擬合（calibration seeds，勿用評估 seeds）
uv run python scripts/fit_action_priors.py --a-runs <A runs> --b-runs <B runs> \
  --out app/simulation_policy/action_priors.json
# calibration 情境庫（tests/fixtures/calibration，與評估庫分開；只跑 seeds 11–13，拒絕 1–5）
uv run python scripts/ab_suite.py --manifest tests/fixtures/calibration/calibration.json --out <cal-out>
# 準備階段校正與語氣範例庫（都只讀 calibration 情境）
uv run python scripts/fit_prep_calibration.py --dirs <cal-out>/<情境> ... --out app/services/stance_calibration.json
uv run python scripts/build_stance_bank.py --lang zh --dirs <cal-out>/<情境> ... --out ../locales/zh_stance_bank.json
# 單一模型的讀出準確度、偏誤與溫度（多模型池）
uv run python scripts/calibrate_model.py --base-url http://127.0.0.1:8002/v1 --model phi-4-mini \
  --prompt-format plain --out phi.json [--write]
```

瀏覽器端到端（需要本機模型服務、`MIROFISH_PROFILE=local` 的後端與前端，不放進預設 CI）：

```bash
npm install && npx playwright install chromium-headless-shell
npm run test:e2e        # E2E_ROUNDS 可調回合數（預設 10），截圖在 tests/e2e/artifacts/
```

2026-09-28 實測（llama.cpp 路徑、golden 種子、10 回合）：通過，耗時 2.3 分鐘。報告頁 6/6 章節都有內容，狀態 Completed，完成後停止輪詢。

這次端到端找到一個 bug：兩個平台都跑完後，模擬程序會保留環境給訪談用，不會結束；runner 卻要等程序結束才發布 COMPLETED，所以 UI 一直停在模擬那一步，報告按鈕不會亮。修正後改成兩個平台都結束、圖譜寫入排空之後就發布（#49）。

## 多模型池（#48）

`MODEL_POOL` 是 OpenAI 相容端點的 JSON 清單；沒設定時就是原本的單一模型，行為不變。

```env
MODEL_POOL=[{"name":"qwen","base_url":"http://127.0.0.1:8000/v1","model":"qwen3.5-4b","roles":["readout","generate","decide"]},{"name":"phi","base_url":"http://127.0.0.1:8002/v1","model":"phi-4-mini","roles":["readout","generate"],"prompt_format":"plain"}]
SYSTEM_ONE_ENSEMBLE=all   # 或逗號分隔的題目 key；未設定＝只用第一個讀出模型
```

- 生成：依 `persona_ref` 的雜湊把每個 agent 固定分派到一個生成模型；呼叫失敗時退回第一個模型。分派記錄在 `decisions.jsonl` 的 `content_model_assigned` 與 `content_metrics` 的 `models`。
- 讀出集成：範圍內的題目由每個讀出模型各答一次，機率取平均後重算答案，仍是零 decode；端點失效時該題改用其餘模型。
- 各模型溫度：`calibrate_model.py --write` 寫進 `app/system_one/calibration.json` 的 `models.<名稱>`。
- 獨立評分：`ab_eval.py --scorer-base-url http://127.0.0.1:8002/v1 --scorer-model phi-4-mini --scorer-prompt-format plain` 會改用另一個模型評貼文立場，模擬用的模型就不必兼任裁判；報告會記錄 `scorer`。
- 非 Qwen 模型的 prompt 格式要實測：Gemma 用 plain 時 score 題的標籤幾乎不在 top-k（覆蓋率 0.0004），要用 chatml；Phi-4-mini 用 plain 較好。
- `ab_eval.py --scorer <名稱>` 直接用 `MODEL_POOL` 裡該名稱的端點當評分器（#54）。
- `calibrate_model.py --with <url>,<模型>[,<格式>]` 可以重複指定，報告的是機率平均後的集成（和 `SYSTEM_ONE_ENSEMBLE` 相同的合併方式），也會列出平均延遲。

### 更小的讀出模型（#54，System One 評估集 205 題）

每個模型都試了 chatml 與 plain 兩種格式，下表列較好的一種。

| 模型 | 格式 | 立場 | 同一實體 | 會互動 | 貼文情緒 | 正面偏誤 | 互動偏誤 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B（現行） | chatml | 0.92 | 0.80 | 0.93 | — | −0.02 | +0.10 |
| Phi-4-mini | plain | 0.78 | 0.97 | 0.73 | — | −0.04 | −0.15 |
| **Qwen3.5-2B** | chatml | 0.64 | 0.83 | 0.83 | 0.97 | +0.04 | −0.04 |
| Qwen3-1.7B | plain | 0.60 | 0.80 | 0.83 | 0.82 | −0.09 | +0.05 |
| Qwen3.5-0.8B | chatml | 0.46 | 0.49 | 0.73 | 0.92 | +0.10 | −0.15 |
| Gemma 3 1B | plain | 0.46 | 0.69 | 0.70 | 0.78 | +0.05 | −0.25 |

1–2B 的模型在立場題上都明顯比 4B 弱，不適合單獨當讀出模型或評分器。

### 10 GB 以內的讀出集成（#54）

VRAM 的量法：啟動或停止一個 llama-server 前後，比較整張卡的用量（各取 3 次的中位數）。Windows 的 `nvidia-smi` 不提供單一程序的用量。

| 服務 | 設定 | VRAM |
| --- | --- | ---: |
| Qwen3.5-4B（主模型） | `-c 65536 -np 8` | 5.4 GB |
| Qwen3.5-2B（第二讀出模型） | `-c 4096 -np 2` | 1.7 GB |
| Phi-4-mini | `-c 4096 -np 2` | 3.2 GB |
| Phi-4-mini | `-c 16384 -np 4` | 4.8 GB |

讀出的 prompt 只有幾百個 token，第二個模型用 4K context 就夠。**Qwen3.5-4B + Qwen3.5-2B 的模型服務共 7.1 GB**，加上桌面約 2.6 GB，整張卡約 9.7 GB，在 10 GB 以內。Qwen + Phi 共 8.6 GB，整張卡約 11.2 GB，超過預算。

| System One 評估集 | 立場 | 同一實體 | 會互動 | 貼文情緒 | 正面偏誤 | 互動偏誤 | 平均延遲 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 0.92 | 0.80 | 0.97 | 0.98 | −0.02 | +0.09 | 615 ms |
| **Qwen3.5-4B + Qwen3.5-2B** | **0.94** | **0.97** | 0.97 | **1.00** | +0.01 | **+0.03** | 1009 ms |
| Qwen3.5-4B + Phi-4-mini（超過 10 GB） | 0.98 | 0.97 | 0.90 | 0.93 | −0.03 | −0.03 | 951 ms |

延遲是在另一個模擬同時使用 GPU 時量的，只適合互相比較。設定方式：

```env
MODEL_POOL=[{"name":"qwen","base_url":"http://127.0.0.1:8000/v1","model":"qwen3.5-4b"},{"name":"q2b","base_url":"http://127.0.0.1:8003/v1","model":"qwen3.5-2b","roles":["readout"]}]
SYSTEM_ONE_ENSEMBLE=all
```

這個集成只在讀出評估集上量過，還沒有跑模擬層級的 A/B。

### 獨立評分器重算收尾基準（#54）

評分器改成 Phi-4-mini（plain），重算 Phase 2 收尾的同一批 runs（6 個情境，每組 5 個 seed）。逐則貼文和 Qwen 評分比較：

- 完全同級只有 16–65%，差一級以內是 49–100%（Northbridge B 是 100%，因為全部都是中立）。
- Phi 幾乎不用「強烈支持」（0–7%，Qwen 是 26–86%），所以整體分數比 Qwen 低 0.03–0.35。

G5 各閘門，兩種評分器對照：

| 閘門 | Qwen 評分 | Phi 評分 |
| --- | ---: | ---: |
| 主要陣營 | 6/6 | 5/6（澄川 A 是 0.49，剛好在邊界） |
| 整體傾向 | 3/6 | **6/6** |
| 走向 | 3/6 | 4/6（golden、澄川仍未過） |
| 角色排序 | 2/6 | 1/6 |

整體傾向的落差大多跟 Qwen 評分器有關：本機模板是用 Qwen 的讀出稽核的，Qwen 也兼任評分器。角色排序與走向的落差換了評分器仍在，是真的差距。

相關 issue：#1（epic）、#2–#13、#19、#26–#28、#34–#36、#43–#49、#51、#52。

# Runs API：一次跑完全流程（#63）

以前要依序呼叫本體、圖譜、建立模擬、準備、啟動、報告六個端點，並各自輪詢。runs API 用一個呼叫在後端依序跑完，進度用 Server-Sent Events 推送，狀態存在 SQLite，後端重啟後查得到，也能從中斷的地方續跑。舊的逐步 API 仍然保留。

## 端點

| 方法 | 路徑 | 說明 |
| --- | --- | --- |
| `POST` | `/api/runs` | 開始一個 run，回傳 `202 {"run_id": ...}` |
| `GET` | `/api/runs` | 最近的 run（`?limit=`，預設 50） |
| `GET` | `/api/runs/<id>` | 狀態、目前階段、各階段產物的 id、失敗原因 |
| `GET` | `/api/runs/<id>/events` | SSE 事件串流；run 結束後關閉 |
| `POST` | `/api/runs/<id>/resume` | 讓失敗或中斷的 run 從第一個沒完成的階段接續 |
| `POST` | `/api/runs/<id>/confirm` | 角色確認完畢，讓暫停中的 run 繼續（見「確認角色」） |

### 開始一個 run

`multipart/form-data` 或 JSON：

| 欄位 | 必填 | 說明 |
| --- | --- | --- |
| `file` 或 `document_text` | 是 | 文件（`.txt`／`.md`／`.pdf`）或直接貼上的文字 |
| `simulation_requirement` | 是 | 模擬需求（要預測什麼） |
| `max_rounds` | 否 | 回合數，預設 24（評估用的回合數） |
| `seeds` | 否 | 跑幾個 seed，預設 3，最多 8 |
| `seed` | 否 | 第一個 seed，預設 1；其餘依序加 1 |
| `confirm_roles` | 否 | `true` 時圖譜建好後暫停，等確認角色 |
| `project_name` | 否 | 專案名稱 |

模式（local／local-hybrid／local-llm）由後端的 `MIROFISH_PROFILE` 決定，會記在 run 的參數裡。`Accept-Language` 也會被記下，各階段用同一種語言。

```bash
curl -F file=@news.txt -F simulation_requirement="預測使用者對漲價的反應" \
     -F max_rounds=24 http://localhost:5001/api/runs
curl -N http://localhost:5001/api/runs/run_xxxxxxxxxxxx/events
```

### 階段

固定的代碼（顯示文字交給前端 i18n），依序執行：

| 代碼 | 做什麼 | 產物 |
| --- | --- | --- |
| `ontology` | 上傳文件並產生本體 | `project_id` |
| `graph` | 建構圖譜 | `graph_id` |
| `prepare` | 建立並準備模擬 | `simulation_id` |
| `simulate` | 每個 seed 一個雙平台模擬 | `seed_simulations`（`seed`、`simulation_id`）、`rounds` |
| `report` | 第一個 seed 的報告 | `report_id` |
| `consistency` | 比較各 seed 的結論 | `consistency`（只有一個 seed 時為 `null`） |

各階段在後端同一個 process 裡呼叫前端原本呼叫的端點（相同的驗證、鎖與背景任務），所以結果和逐步操作 UI 相同。兩個差異：run 預設**不**開啟圖譜記憶回寫（多個 seed 同時回寫同一張圖會互相干擾，metrics 報告也不需要）；模擬一律 `force` 重新開始。

### 事件

每個事件有遞增的 `id`，SSE 的 `event:` 是種類，`data:` 是 JSON（`id`、`ts`、`stage`、`kind`、`payload`）：

| 種類 | payload |
| --- | --- |
| `run_start` | `resume_from`：從哪個階段開始 |
| `stage_start` | — |
| `progress` | `progress`、`message`（來自各階段的狀態端點） |
| `stage_done` | `artifacts`、`seconds` |
| `run_done` | `artifacts` |
| `run_failed` | `reason` |
| `awaiting_confirmation` | `graph_id`：暫停等確認角色 |
| `interrupted` | 後端重啟時 run 還在進行 |

斷線重連時帶 `Last-Event-ID`（或 `?after=<id>`），會從下一個事件接著送。沒有新事件時每 15 秒送一個心跳註解。

### 多個 seed

`prepare` 只跑一次。第一個 seed 用準備好的模擬，其餘 seed 各用一份複製（`POST /api/simulation/copy`：同樣的角色與設定，新的 `simulation_id`），只差在 agent 啟用順序的 seed。

同時跑幾個由主機可用記憶體決定：(可用記憶體 − `RUNS_MEMORY_RESERVE_MB`) ÷ `RUNS_SIM_MEMORY_MB`，至少 1，其餘排隊。模型服務是共用的，seed 越多，每回合越慢。進度事件的 `message` 是所有 seed 的回合總和，例如 `round 30/72 over 3 seeds, 1 done`。續跑時沿用已複製的模擬，已跑完的 seed 不重跑。

### 一致性

`consistency` 階段對每個 seed 的貼文做立場評分（和 `scripts/scan.py` 相同，用本機模型的讀出），回傳：

| 欄位 | 說明 |
| --- | --- |
| `main_camp` | 多數 seed 的主要陣營（`oppose`／`neutral`／`support`）與一致的 seed 數 |
| `tendency`、`trend` | 整體傾向與走向，跨 seed 的平均與標準差 |
| `ranking` | 平均最支持、最反對的角色；`stability` 是兩半 seed 角色排序的 Spearman |
| `confirm` | 需要用 `local-llm` 確認的原因；空的代表 seed 間一致 |

seed 之間一致，只代表這條路徑自己穩定，不代表和 LLM 路徑一致（見 docs/local-first.md 的 G5）。

### 確認角色

帶 `confirm_roles=true` 時，圖譜建好後 run 狀態變成 `awaiting_confirmation`，送出 `awaiting_confirmation` 事件，事件串流關閉。這時可以用 `GET`／`POST /api/graph/<graph_id>/roles`（#64）查看、修改角色，再 `POST /api/runs/<id>/confirm` 繼續；重新連上事件串流即可接著收。暫停中的 run 不受後端重啟影響，`resume` 不適用於暫停中的 run。

### 失敗與續跑

任何一個端點回傳錯誤（非 2xx 或 `success: false`），run 就停在該階段，狀態為 `failed`，`error` 是端點的錯誤訊息（例如 #62 的「模型服務無回應」）。後端重啟時還在進行的 run 標為 `interrupted`。兩種都可以 `POST /api/runs/<id>/resume`：已完成的階段會跳過，並沿用它們的產物 id。`ontology` 階段若在建立專案後被切斷，續跑時會沿用已產生本體的專案，並刪掉沒完成的專案。

狀態存在 `backend/uploads/runs/runs.sqlite`（可用 `RUNS_DB_PATH` 改位置），文件存在 `backend/uploads/runs/<run_id>/`。

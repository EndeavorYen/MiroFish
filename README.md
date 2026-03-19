# MiroFish

> Fork from [666ghj/MiroFish](https://github.com/666ghj/MiroFish) — 原專案由盛大集團孵化，模擬引擎由 [OASIS](https://github.com/camel-ai/oasis) 驅動。

簡潔通用的群體智能引擎 — 透過 LLM 驅動的多 Agent 模擬社群行為，產生預測報告。

## 專案概述

**MiroFish** 是一款基於多智能體技術的 AI 預測引擎。上傳種子資料（新聞、報告、小說等），用自然語言描述預測需求，MiroFish 會自動建構數位平行世界，讓上千個具備獨立人格與記憶的 Agent 在社群平台上自由互動，最終生成一份詳盡的預測報告。

### 本 Fork 的主要改動

- **去 Zep 化**：移除 Zep Cloud 依賴，改用 NetworkX + SQLite 混合架構（本地圖譜 + 全文檢索）
- **零外部服務**：知識圖譜完全本地化，資料持久化至 SQLite，重啟不遺失
- **中文介面**：前端支援繁體中文 / 簡體中文 / 英文切換

## 系統截圖

<div align="center">
<table>
<tr>
<td><img src="./static/image/Screenshot/運行截圖1.png" alt="截圖1" width="100%"/></td>
<td><img src="./static/image/Screenshot/運行截圖2.png" alt="截圖2" width="100%"/></td>
</tr>
<tr>
<td><img src="./static/image/Screenshot/運行截圖3.png" alt="截圖3" width="100%"/></td>
<td><img src="./static/image/Screenshot/運行截圖4.png" alt="截圖4" width="100%"/></td>
</tr>
</table>
</div>

## 工作流程

```
種子資料上傳 → 知識圖譜建構 → 環境與角色設定 → 社群模擬運行 → 報告生成與互動
```

1. **圖譜建構** — 上傳文字資料，LLM 自動抽取實體與關係，建構知識圖譜
2. **環境設定** — 根據圖譜生成 Agent 角色人設，配置模擬參數（平台、輪數等）
3. **模擬運行** — Twitter / Reddit 雙平台並行模擬，Agent 自主發文、互動、搜尋
4. **報告生成** — ReportAgent 透過圖譜搜尋工具分析模擬結果，產出預測報告
5. **深度互動** — 與 ReportAgent 或模擬世界中的任意 Agent 進行對話

## 快速開始

### 前置需求

| 工具 | 版本 | 用途 | 檢查指令 |
|------|------|------|---------|
| **Node.js** | 18+ | 前端執行環境 | `node -v` |
| **Python** | 3.11 ~ 3.12 | 後端執行環境 | `python --version` |
| **uv** | 最新版 | Python 套件管理 | `uv --version` |

### 1. 設定環境變數

```bash
cp .env.example .env
```

編輯 `.env`，填入 LLM API 資訊：

```env
# LLM API 設定（支援 OpenAI SDK 格式的任意 LLM API）
# 推薦使用阿里百煉平台 qwen-plus：https://bailian.console.aliyun.com/
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen-plus
```

<details>
<summary>使用本地 Ollama（免費替代方案）</summary>

```bash
# 啟動 Ollama 服務
ollama serve

# 拉取模型（擇一）
ollama pull qwen2.5:7b-instruct   # 輕量
ollama pull qwen3.5:9b             # 推薦，支援思考模式
```

```env
# .env 設定
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_NAME=qwen3.5:9b

# 思考模式（預設 false，適用 qwen3.5 等支援推理的模型）
# 開啟後報告生成品質更高，但 token 消耗與回應時間會增加
# LLM_THINK=true
```

</details>

> **注意**：本 Fork 已移除 Zep Cloud 依賴，不需要設定 `ZEP_API_KEY`。

### 2. 安裝相依套件

```bash
# 一鍵安裝（根目錄 + 前端 + 後端）
npm run setup:all
```

或分步安裝：

```bash
npm run setup           # Node 相依（根目錄 + 前端）
npm run setup:backend   # Python 相依（後端，自動建立 venv）
```

### 3. 啟動服務

```bash
# 同時啟動前後端
npm run dev
```

啟動後開啟瀏覽器：

| 服務 | 位址 |
|------|------|
| 前端介面 | http://localhost:3000 |
| 後端 API | http://localhost:5001 |

單獨啟動：

```bash
npm run backend    # 僅後端
npm run frontend   # 僅前端
```

### Docker 部署

```bash
cp .env.example .env    # 設定環境變數
docker compose up -d    # 啟動容器
```

預設使用 `.env` 並映射 `3000`（前端）/ `5001`（後端）埠號。

## 使用指南

### Step 1：上傳種子資料

在首頁上傳文字資料（支援直接貼上文字或上傳檔案）。系統會透過 LLM 抽取實體與關係，建構知識圖譜。

### Step 2：設定模擬環境

- 選擇模擬平台（Twitter / Reddit）
- 設定模擬輪數（建議初次嘗試 20~40 輪）
- 用自然語言描述你的預測需求
- 系統會自動生成 Agent 角色人設與模擬配置

### Step 3：執行模擬

點擊開始後，Agent 將在虛擬社群平台上自主互動。你可以即時觀看模擬過程中 Agent 的發文、轉發、評論等行為。

### Step 4：查看報告

模擬結束後，ReportAgent 會分析所有 Agent 的互動記錄，產出一份結構化的預測報告。你也可以繼續與 ReportAgent 對話，深入探討模擬結果。

## 技術架構

```
frontend/          Vue 3 + Vite 前端
backend/
  app/
    api/           Flask REST API
    services/
      graph_store.py              資料模型與 Protocol 定義
      networkx_graph_store.py     NetworkX + SQLite 圖譜儲存（啟動時自動從 DB 恢復）
      graph_builder.py            LLM 實體抽取 + 圖譜建構
      ontology_generator.py       本體論生成（實體/關係類型設計）
      simulation_config_generator.py  模擬配置生成
      oasis_profile_generator.py  Agent 角色人設生成
      graph_memory_updater.py     模擬活動 → 圖譜邊（零 LLM 成本）
      search_tools.py             搜尋工具（QuickSearch / InsightForge）
      report_agent.py             報告生成 Agent
      simulation_runner.py        OASIS 模擬執行器
    utils/
      llm_client.py               統一 LLM 呼叫入口
  uploads/
    graph_store.db                知識圖譜 SQLite 資料庫
    simulations/                  模擬狀態與資料（檔案系統）
    projects/                     專案與上傳檔案
    reports/                      分析報告
```

## 致謝

- 原專案 [666ghj/MiroFish](https://github.com/666ghj/MiroFish)，由盛大集團戰略支持與孵化
- 模擬引擎 [OASIS](https://github.com/camel-ai/oasis)，由 CAMEL-AI 團隊開源貢獻

## 授權

[AGPL-3.0](LICENSE)

# MiroFish

> Forked from [666ghj/MiroFish](https://github.com/666ghj/MiroFish) — originally incubated by Shanda Group, simulation engine powered by [OASIS](https://github.com/camel-ai/oasis).

A simple and universal swarm intelligence engine — using LLM-driven multi-agent simulation of social behavior to generate prediction reports.

## Overview

**MiroFish** is an AI prediction engine powered by multi-agent technology. Upload seed materials (news, reports, novels, etc.), describe your prediction needs in natural language, and MiroFish will automatically construct a parallel digital world where thousands of agents with independent personalities and memories interact freely on social platforms, ultimately generating a detailed prediction report.

### Key Changes in This Fork

- **De-Zep**: Removed Zep Cloud dependency, replaced with NetworkX + SQLite hybrid architecture (local graph + full-text search)
- **Zero External Services**: Memory graph is fully local — no third-party cloud services required
- **i18n**: Frontend supports Traditional Chinese / English switching

## Workflow

```
Seed Upload → Knowledge Graph → Environment Setup → Simulation → Report & Interaction
```

1. **Graph Building** — Upload text, LLM extracts entities and relations to build a knowledge graph
2. **Environment Setup** — Generate agent personas from the graph, configure simulation parameters
3. **Simulation** — Twitter / Reddit dual-platform parallel simulation with autonomous agent interactions
4. **Report Generation** — ReportAgent analyzes simulation results via graph search tools
5. **Deep Interaction** — Chat with ReportAgent or any agent in the simulated world

## Quick Start

### Prerequisites

| Tool | Version | Purpose | Check |
|------|---------|---------|-------|
| **Node.js** | 18+ | Frontend runtime | `node -v` |
| **Python** | 3.11 ~ 3.12 | Backend runtime | `python --version` |
| **uv** | Latest | Python package manager | `uv --version` |

### 1. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your LLM API credentials:

```env
# LLM API (any OpenAI SDK-compatible API)
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen-plus
```

<details>
<summary>Using local Ollama (free alternative)</summary>

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

```env
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_NAME=qwen2.5:7b-instruct
```

</details>

> **Note**: This fork has removed Zep Cloud — no `ZEP_API_KEY` needed.

### 2. Install Dependencies

```bash
npm run setup:all
```

### 3. Start Services

```bash
npm run dev
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:5001 |

### Docker

```bash
cp .env.example .env
docker compose up -d
```

## Acknowledgments

- Original project [666ghj/MiroFish](https://github.com/666ghj/MiroFish), strategically supported by Shanda Group
- Simulation engine [OASIS](https://github.com/camel-ai/oasis) by the CAMEL-AI team

## License

[AGPL-3.0](LICENSE)

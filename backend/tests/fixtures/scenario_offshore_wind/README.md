# 第二情境：東濱一號離岸風場

用來檢查本機路徑（System One 決策、分層內容、模板準備，以及在 golden 上擬合的 action priors）是否能泛化到 golden 以外的情境。`news_seed.txt` 與 `tests/fixtures/holdout5_extraction` 相同；`simulation_requirement.txt` 在跑任何模擬前寫定。

```bash
uv run python scripts/golden_pipeline.py prepare --work <dir> --prep-mode llm --fixture tests/fixtures/scenario_offshore_wind
```

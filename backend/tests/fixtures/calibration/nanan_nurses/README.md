# nanan_nurses

Calibration 情境，不在評估情境庫（`tests/fixtures/scenarios/suite.json`）裡，只跑 calibration seeds 11–13。用來擬合 action priors、準備階段的立場校正與語氣範例庫。

來源：tests/fixtures/holdout3_extraction 的種子文（實體抽取 holdout）。地名、機構與人物皆為虛構。

- 領域：醫療
- 語言：中文
- 預期立場結構：兩極對立
- fixture digest：`53d61677907306319d9d4e7b2830c3d59ea3f8760ed060129394e03a21c34cfa`

digest 只涵蓋 `news_seed.txt` 與 `simulation_requirement.txt`（換行先正規化成 LF）。

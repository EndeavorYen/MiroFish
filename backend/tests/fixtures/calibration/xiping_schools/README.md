# xiping_schools

Calibration 情境，不在評估情境庫（`tests/fixtures/scenarios/suite.json`）裡，只跑 calibration seeds 11–13。用來擬合 action priors、準備階段的立場校正與語氣範例庫。

來源：tests/fixtures/holdout4_extraction 的種子文（實體抽取 holdout）。地名、機構與人物皆為虛構。

- 領域：教育
- 語言：中文
- 預期立場結構：兩極對立
- fixture digest：`df09475ac617c5fda06a50a47846f546935d4aca7bdf9be2ce3b97de7c48b231`

digest 只涵蓋 `news_seed.txt` 與 `simulation_requirement.txt`（換行先正規化成 LF）。

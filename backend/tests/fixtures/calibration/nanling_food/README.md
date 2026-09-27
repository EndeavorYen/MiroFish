# nanling_food

Calibration 情境，不在評估情境庫（`tests/fixtures/scenarios/suite.json`）裡，只跑 calibration seeds 11–13。用來擬合 action priors、準備階段的立場校正與語氣範例庫。

來源：tests/fixtures/holdout_extraction 的種子文（實體抽取 holdout）。地名、機構與人物皆為虛構。

- 領域：消費爭議
- 語言：中文
- 預期立場結構：兩極對立
- fixture digest：`07065a12ba696c035f06871382b3a2bf4865a9aef92ff343439e8b66da95d371`

digest 只涵蓋 `news_seed.txt` 與 `simulation_requirement.txt`（換行先正規化成 LF）。

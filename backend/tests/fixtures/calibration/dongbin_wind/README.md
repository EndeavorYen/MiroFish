# dongbin_wind

Calibration 情境，不在評估情境庫（`tests/fixtures/scenarios/suite.json`）裡，只跑 calibration seeds 11–13。用來擬合 action priors、準備階段的立場校正與語氣範例庫。

來源：tests/fixtures/scenario_offshore_wind（#41 的第二情境）。地名、機構與人物皆為虛構。

- 領域：能源
- 語言：中文
- 預期立場結構：兩極對立
- fixture digest：`326870a3f1c65fb3aa9592fdb4b91d2628f233ad5ed6ba94a673219bfa4c94b5`

digest 只涵蓋 `news_seed.txt` 與 `simulation_requirement.txt`（換行先正規化成 LF）。

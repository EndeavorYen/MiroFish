# System One eval set

`system_one_eval.jsonl` — 205 synthetic Traditional Chinese items for the
decisions a social simulation asks System One to make.

| category | type | n | label |
| --- | --- | --- | --- |
| `post_sentiment` | choice (positive / neutral / negative) | 60 | option key |
| `post_stance` | choice (support / oppose / neutral / other) | 50 | option key |
| `will_engage` | noul | 30 | bool |
| `same_entity` | noul | 35 | bool |
| `emotion_intensity` | score (3 levels) | 30 | 0-based level |

## Label source

One-time strong-model labelling: every item and its gold label were written
together by Claude Opus 5.5 (the implementer of #4) in a single pass, then
re-read once by the same model. No human has spot-checked the labels yet.
All text is invented; no copyrighted articles are used. Items were written to
be unambiguous, so accuracy here is an upper bound for real simulation feeds.

Row shape:

```json
{"id": "sent-001", "category": "post_sentiment", "type": "choice",
 "state": "...", "instructions": "...", "criteria": {"positive": "..."},
 "label": "positive"}
```

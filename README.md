# koku

Turkish fragrance assistant: distilled small LM, fine-tuned retriever, in-browser demo.

## Data

Fragrantica.com Fragrance Dataset (Kaggle, olgagmiufana1, v3, 2024-09-21), CC BY-NC-SA 4.0. Non-commercial; derived models and data are shared under the same license.

## Pipeline

| Step | Script | Output |
|---|---|---|
| Download | `uv run --with kaggle kaggle datasets download olgagmiufana1/fragrantica-com-fragrance-dataset -p data --unzip` | `data/fra_cleaned.csv` |
| Profile | `uv run --with pandas python profile_data.py` | stdout |
| Clean | `uv run --with pandas python clean.py` | `data/perfumes.jsonl` (24,063 perfumes) |

`fra_cleaned.csv` is `;`-separated, decimal comma, cp1252-encoded (`0x99` = ™, `0x92` = ’). Cleaning lowercases notes, strips ®/™, dedupes notes per tier, and title-cases name/brand slugs.

## Evaluation set

`eval/queries.jsonl` holds 200 Turkish search queries. Each query carries machine-checkable constraints (accords, notes, gender) instead of hand-picked answers, so relevance is reproducible and auditable:

```bash
python eval/label.py      # constraints -> eval/qrels.jsonl + eval/review.md (human review sheet)
python eval/evaluate.py --baseline        # popularity baseline
python eval/evaluate.py runs/<run>.jsonl  # any retriever
```

Corpus: 13,718 perfumes with at least 100 votes. Relevant perfumes per query: min 5, median 192, max 791.
Metrics at k=10: precision, nDCG, MRR. Recall is not reported because most queries have far more than 10 relevant perfumes.

| Run | P@10 | nDCG@10 | MRR@10 |
|---|---|---|---|
| Popularity (same top-voted list for every query) | 0.019 | 0.017 | 0.039 |

Known limits: queries were drafted with an LLM and are pending human review (`eval/review.md`); constraints only see the top-5 accords and listed notes, so a correct but unlisted perfume counts as a miss.

# Topic-Model Experiment Data — Structure & Query Guide

This document describes every dataset used in the BOE topic-model studies, how the
pieces fit together, where each is complete or ragged, and how to locate and query
it. It covers three datasets:

1. **Existing (non-BOE) topic models** — the published baseline.
2. **Main BOE experiment** — Bag-of-Embeddings topic models (the experimental arm).
3. **BOE chunk-size experiment (CSE)** — a self-contained study of `target_chunk_count`.

Last verified against the database: 2026-05-31 (CSE metrics added & verified 2026-06-06).
All counts below are live unless stated.

---

## 0. Database access

```bash
# from the repo root (/home/darren/Documents/PhD/tm_framework)
. .env                        # exports DB_URI
psql $DB_URI -c "SELECT 1"    # all data is in schema `pipeline`
```

All tables live in the `pipeline` schema. `\dt pipeline.*` lists them.
Every result table has a `soft_delete BOOLEAN` flag — **always filter `soft_delete = FALSE`**
(currently 0 rows are soft-deleted anywhere, but filter anyway for safety).

Shared experimental grid (identical across all three datasets unless noted):

| Axis | Values |
|---|---|
| **Corpora** (10) | `battery-abstracts`, `goodreads-bookgenres`, `imdb_reviews`, `newsgroups`, `patent-classification`, `pubmed-multilabel`, `t2-ragbench-convfinqa`, `trec_questions`, `twitter-financial-news`, `wikipedia_sample` |
| **num_topics** (5) | 10, 20, 50, 100, 200 |
| **Metrics** (4) | `NPMI`, `WEPS`, `WECS`, `ISH` — analysis code flips ISH→**NISH** (negated) so higher = better for all four |
| **Problem instance** | `(corpus, num_topics)` → 50 instances total |

All metric values are clean — no NULL / NaN / Inf rows.

---

## 1. Existing (non-BOE) topic models — the baseline

The published baseline; fully balanced, no gaps.

**Tables & join:**
```sql
SELECT c.name AS corpus, m.name AS model, r.num_topics, p.metric_name, p.metric_value
FROM pipeline.topic_model_performance p
JOIN pipeline.topic_model_corpus_result r ON r.id = p.topic_model_corpus_result_id
JOIN pipeline.topic_model m               ON m.id = r.topic_model_id
JOIN pipeline.corpus c                    ON c.id = r.corpus_id      -- corpus is an ID join here
WHERE r.soft_delete = FALSE;
```
Note: this side joins `corpus` by **`corpus_id`** (the BOE sides use a `corpus_name` string instead).

**Dimensions:**
- **11 models:** `BERTopic`, `BERTopic_sbert`, `CombinedTM`, `CombinedTM_sbert`, `GMM`, `KeyNMF`, `LDA`, `NMF`, `SemanticSignalSeparation`, `ZeroShotTM`, `ZeroShotTM_sbert`.
  - 6 share a name with a BOE family (BERTopic, CombinedTM, GMM, KeyNMF, SemanticSignalSeparation, ZeroShotTM).
  - 5 are baseline-only: `BERTopic_sbert`, `CombinedTM_sbert`, `ZeroShotTM_sbert`, `LDA`, `NMF`.
- **10 repeats** per `(corpus, num_topics)` cell. *(≠ BOE's 5 — see gotchas.)*

**Volume:** 11 models × 500 results = 5,500 result rows; × 4 metrics = **22,000 metric rows**.
Fully complete — every model has all 10 corpora × 5 topic settings × 10 repeats.

---

## 2. Main BOE experiment — Bag-of-Embeddings topic models

**Tables & join:**
```sql
SELECT r.corpus_name, tm.name AS family, r.num_topics,
       r.target_dims, r.padding_method, r.target_chunk_count,
       p.metric_name, p.metric_value
FROM pipeline.boe_topic_model_performance p
JOIN pipeline.boe_topic_model_corpus_result r ON r.id = p.boe_topic_model_corpus_result_id
JOIN pipeline.boe_topic_model tm              ON tm.id = r.topic_model_id
WHERE r.soft_delete = FALSE;
```
Supporting tables (not usually needed for analysis): `boe_document_embedding`,
`boe_word_embedding`, `boe_embedding` (chunk embeddings, shared upstream),
`boe_embedding_reduced`, `boe_embedding_sparse`.

**Volume:** 7,426 result rows / **29,704 metric rows** (= ×4 metrics).

### Factors

| Factor | Values | Notes |
|---|---|---|
| **Embedding model** | `all-MiniLM-L6-v2` only | column `source_model_name`; not a slicing axis |
| **Topic-model family** (6) | BERTopic, CombinedTM, GMM, KeyNMF, SemanticSignalSeparation, ZeroShotTM | from `boe_topic_model.name` |
| **target_dims** (4) | `0`, `20`, `50`, `100` | `0` = unreduced (`algorithm='none'`); `20/50/100` = UMAP-reduced (`algorithm='umap'`) |
| **padding_method** (2) | `noise_only` (primary), `knn_mean` (sparse secondary) | how documents are padded to `target_chunk_count` chunks |
| **target_chunk_count** | 1, 2, 4, 6, 9 | **NOT a free axis — fixed per corpus** (see below) |

**`target_chunk_count` is a per-corpus property**, not an independently swept factor:

| chunk_count | corpora |
|---|---|
| 1 | patent-classification, trec_questions, twitter-financial-news |
| 2 | battery-abstracts, goodreads-bookgenres, newsgroups, pubmed-multilabel |
| 4 | imdb_reviews |
| 6 | t2-ragbench-convfinqa |
| 9 | wikipedia_sample |

The BOE document embedding is the **stacked** vector of `target_chunk_count` chunk
embeddings, so its dimensionality is `target_chunk_count × target_dims`
(e.g. d50 on wikipedia = 9×50 = 450; d0 = chunk_count×384). This matters for GMM (§4).

**Repeats:** 5 per `(family, target_dims, padding, corpus, num_topics)` cell.
A full cell = 10 corpora × 5 topics × 5 repeats = **250 result rows**.

### Completeness cube (result rows; **250 = a complete cell**)

| Family | padding | d0 | d20 | d50 | d100 |
|---|---|---|---|---|---|
| BERTopic | noise_only | 250 | 250 | 250 | 250 |
| CombinedTM | noise_only | 250 | 250 | 250 | 250 |
| KeyNMF | noise_only | 250 | 250 | 250 | 250 |
| SemanticSignalSeparation | noise_only | 250 | 250 | 250 | 250 |
| ZeroShotTM | noise_only | 250 | 250 | 250 | 250 |
| **GMM** | noise_only | 250 | 125 | 200 | 235 |
| BERTopic | knn_mean | 0 | 0 | 5 | 150 |
| CombinedTM | knn_mean | 0 | 0 | 150 | 150 |
| KeyNMF | knn_mean | 0 | 0 | 150 | 150 |
| SemanticSignalSeparation | knn_mean | 0 | 0 | 150 | 150 |
| ZeroShotTM | knn_mean | 0 | 0 | 150 | 150 |
| GMM | knn_mean | 0 | 0 | 120 | 141 |

- **`noise_only` is the clean arm:** 5 of 6 families are fully complete across all four dims (1,000 rows each). **Only GMM is ragged** (see §4).
- **`knn_mean` is sparse:** only d50/d100, never reaches a full 250, BERTopic-d50 is a 5-row stub. Treat as a secondary/exploratory padding comparison, not a balanced factorial.

---

## 3. BOE chunk-size experiment (CSE)

**Purpose:** isolate the effect of `target_chunk_count` (how many chunk slots each
document is padded/truncated to) on downstream topic models and on runtime, on a
**fixed, unreduced embedding slice**. Fully self-contained — owns its own
`boe_cse_*` tables and never touches the production `boe_*` tables. The only shared
inputs are upstream chunk embeddings and corpus vocabulary.

**Fixed embedding slice:** `source_model_name='all-MiniLM-L6-v2'`, `algorithm='none'`,
`target_dims=0` (unreduced), `padding_method='noise_only'`.

**Tables & join:**
```sql
SELECT r.corpus_name, tm.name AS family, r.num_topics, r.target_chunk_count,
       p.metric_name, p.metric_value->>'score' AS score
FROM pipeline.boe_cse_topic_model_performance p
JOIN pipeline.boe_cse_topic_model_corpus_result r ON r.id = p.boe_cse_topic_model_corpus_result_id
JOIN pipeline.boe_cse_topic_model tm              ON tm.id = r.topic_model_id
WHERE r.soft_delete = FALSE;
```
Other tables: `boe_cse_document_embedding` (119,408 rows, has `chunk_count` &
`padded_to`), `boe_cse_word_embedding` (21,192), `boe_cse_timing_result` (1,520).

**Metrics:** `pipeline.boe_cse_topic_model_performance` mirrors
`boe_topic_model_performance` (`metric_name`, `metric_value` jsonb `{"score": …}`,
unique per `(result_id, metric_name)`). **11,400 metric rows** = 2,850 results × 4
metrics, fully complete, no NULL/NaN/Inf. Computed by
`boe_chunk_size_experiment/metrics.py` (added 2026-06-06) using the **same shared
implementations** (`pipelines/performance_metrics/`) as the main BOE and baseline
metrics — NPMI from corpus stats, WEPS/WECS/ISH from the shared
`vocabulary_word_embeddings` (FastText), *not* from `boe_cse_word_embedding` (those
are topic-model inputs only). Values are therefore directly comparable across all
three datasets; verified: CSE means at each corpus's natural chunk count match the
main-BOE d0/noise_only means to within ~0.003 on every metric.

### Factors

| Factor | Values |
|---|---|
| **Corpora** | original sweep used 4 corpora (where chunk count varies); the **2026-06-06 BOE@1 extension added `target_chunk_count=1` for ALL 10 corpora** |
| **target_chunk_count** (swept, per corpus) | imdb `{1,2,4}`, patent `{1,2}`, convfinqa `{1,2,4,6}`, wikipedia `{1,3,6,9}`; the other 6 corpora have `{1}` only |
| **Families** (6) | same 6 as main BOE |
| **num_topics** (5) | 10, 20, 50, 100, 200 |
| **Repeats** | 5 (`TARGET_RESULTS`) |

The swept chunk counts (≥2) differ from each corpus's *natural* value used in the
main experiment (e.g. wikipedia is fixed at 9 in the main experiment but sweeps
{3,6,9} here). The **`target_chunk_count=1` slice ("BOE@1")** is the single-chunk
degenerate case — one embedding per document, ≈ the existing baselines — added for
investigation 3's follow-up (see §7); present for all 10 corpora.

**Volume & completeness — fully complete, including GMM** (GMM is complete because
on unreduced embeddings the dimensionality is `chunk_count×384 ≥ 384`, so the
`num_topics ≤ dim` guard (§4) never bites for `num_topics ≤ 200`). Each
(corpus, chunk) cell = 6 families × 5 topics × 5 repeats = 150 results:
- chunk=1 (BOE@1): 10 corpora → **1,500 results**.
- chunk≥2 sweep: 9 combos (imdb {2,4}; patent {2}; convfinqa {2,4,6}; wikipedia
  {3,6,9}) → **1,350 results**.
- **Total 2,850 results.** Verified live: 2,850 results / 11,400 metric rows, 0 missing.

**Timing** (`boe_cse_timing_result.pipeline_stage`): `boe_doc_embedding`,
`boe_word_embedding`, `boe_topic_model` (with `repeat_number`, `duration_seconds`,
`num_topics`, `target_chunk_count`).

### 3a. Normal-word-embedding ablation (`boe_nwe_*`)

A small sibling experiment (`src/python/pipelines/boe_normal_we_experiment/`, added
2026-06-07) isolating BOE's **derived** word embeddings. Trains **only KeyNMF and
SemanticSignalSeparation** (the only two families that consume word embeddings) on
the **identical** first-chunk all-MiniLM-L6-v2 document embeddings as BOE@1 (reuses
`boe_cse_document_embedding` @ `target_chunk_count=1`) but with **normal** word
embeddings — each vocabulary word encoded *directly* with all-MiniLM-L6-v2 (384-d,
in `boe_nwe_word_embedding`), instead of BOE's TF-IDF+Ridge regression-derived
vectors. Tables: `boe_nwe_word_embedding`, `boe_nwe_topic_model_corpus_result`
(string `family` column, no registry), `boe_nwe_topic_model_performance`.
**Volume:** 2 families × 10 corpora × 5 topics × 5 repeats = **500 results / 2,000
metric rows**, complete & clean. Because BOE@1 and NWE share a byte-identical doc
representation, `(BOE@1 − NWE)` cleanly isolates the derived-vs-normal
word-embedding effect (see investigation 3 §3.9).

---

## 4. The GMM coverage constraint (applies to the main BOE experiment)

GMM is the **only** family with structural gaps. The runner
(`boe_06_topic_models/experiment_runner.py:357`) explicitly skips a run when
`num_topics > embeddings.shape[1]`, and the BOE doc-embedding dimensionality is
`target_chunk_count × target_dims`.

**Exact, data-verified rule (0 violations):**
> A GMM row exists **iff `num_topics ≤ target_chunk_count × target_dims`**.

Consequences:
- For **1-chunk** corpora (patent/trec/twitter) this is just `num_topics ≤ target_dims`.
- **High-chunk** corpora survive larger `num_topics` (wikipedia ×9, convfinqa ×6).
- At **d0** the bound is `chunk_count×384` — never binding → GMM complete at d0 and throughout CSE.
- Failures are **deterministic and all-or-nothing** (every cell is exactly 5 or 0 repeats); re-running recovers nothing. These are the only runs GMM can produce under the current model config.
- Underlying model: `turftopic.GMM` → `BayesianGaussianMixture(covariance_type='full')` with internal `PCA(num_topics)` (`src/python/pipelines/topic_models/gmm.py`). Gaps could only be filled by changing the model (lower covariance_type, higher `reg_covar`, etc.), which would break comparability.

GMM completeness by `(num_topics, target_dims)` in `noise_only` (instances present, max 50):

| target_dims | nt=10 | nt=20 | nt=50 | nt=100 | nt=200 |
|---|---|---|---|---|---|
| d0 (=tcc×384) | 50 | 50 | 50 | 50 | 50 |
| d20 | 50 | 50 | 15 | 10 | 0 |
| d50 | 50 | 50 | 50 | 35 | 15 |
| d100 | 50 | 50 | 50 | 50 | 35 |

---

## 5. Cross-cutting gotchas

1. **Repeat asymmetry:** existing models have **10** repeats; BOE (main & CSE) have **5**. Fine for mean comparisons; biases any variance/stddev comparison.
2. **GMM raggedness** (§4) is the only hole in the otherwise-complete `noise_only` block of the main experiment.
3. **`target_dims = 0` is the unreduced ablation**, not "reduce to 0 dims" — `algorithm='none'`. Keep it out of "which reduction level is best" comparisons; use it for a no-reduction-vs-reduction ablation.
4. **`target_chunk_count ≡ corpus`** in the main experiment, so padding-method effects are unidentifiable on the three 1-chunk corpora (nothing to pad).
5. **`knn_mean` is sparse** (main experiment): only d50/d100, never a full cell.
6. **Join differences:** existing side joins corpus by `corpus_id`; both BOE sides carry `corpus_name` as a string column (no corpus-table join needed).
7. **Algorithm column wording:** result tables use `algorithm IN ('none','umap')`; the embedding tables may also store `'pca'`. Slice on `target_dims` (0 vs >0), not on the algorithm string, to be safe.

---

## 6. Ready-to-use slices

```sql
-- Main BOE, the clean balanced factorial (exclude ragged GMM):
WHERE soft_delete = FALSE AND padding_method = 'noise_only'
  AND target_dims IN (20,50,100)
  -- and family <> 'GMM', or handle GMM's partial n explicitly

-- Main BOE, padding-method comparison at a fixed dim:
WHERE soft_delete = FALSE AND target_dims = 100
  AND padding_method IN ('noise_only','knn_mean')

-- Main BOE, no-reduction ablation:
WHERE soft_delete = FALSE AND padding_method = 'noise_only'
  AND target_dims IN (0,100)

-- GMM-safe region (no missing cells):  num_topics <= target_chunk_count * target_dims
```

---

## 7. Where the code lives

| Component | Path |
|---|---|
| Main BOE topic-model runner (GMM guard at :357) | `src/python/pipelines/boe_06_topic_models/experiment_runner.py` |
| Topic-model implementations (incl. `gmm.py`) | `src/python/pipelines/topic_models/` |
| Existing-model metrics / paper code | `src/python/pipelines/performance_metrics/` |
| Chunk-size experiment | `src/python/pipelines/boe_chunk_size_experiment/` (`common.py`, `doc_embedding.py`, `word_embedding.py`, `experiment_runner.py`, `metrics.py`, `schema.sql`) |
| Investigation 1 (complementarity; padding axis) | `src/python/pipelines/boe_performance_comparison_investigation_1/` |
| Investigation 2 (target_dims axis) | `src/python/pipelines/boe_performance_comparison_investigation_2/` |
| Investigation 3 (chunk-count axis, on CSE data) | `src/python/pipelines/boe_chunk_size_investigation/` (`prompt.md`; scripts + `report.pdf` in `ignore/investigation/`) |
| Audit of investigations 1 & 2 | `…/boe_performance_comparison_investigation_{1,2}/AUDIT_FINDINGS.md` |

> Investigation scripts load with only `soft_delete=FALSE` (+ `noise_only` in inv2)
> and **no `target_dims` filter** — so reruns silently mix d0/d20/d50/d100 into one
> label. Pin the intended slice explicitly (see §6 and the AUDIT_FINDINGS files).

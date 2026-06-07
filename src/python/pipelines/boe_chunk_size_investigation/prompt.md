# Investigation 3: Effect of target_chunk_count on BOE topic-model performance

## Motivation

Investigation 1 (Analysis 5) attempted to study how document chunk count affects the
BOE-vs-traditional performance delta, but in the **main BOE experiment**
`target_chunk_count` is fixed per corpus (patent/trec/twitter=1, battery/goodreads/
newsgroups/pubmed=2, imdb=4, convfinqa=6, wikipedia=9), so chunk count is fully
confounded with corpus identity — no causal statement was possible.

The **BOE chunk-size experiment (CSE)** removes this confound: it sweeps
`target_chunk_count` *within* four corpora on a fixed, unreduced embedding slice
(`all-MiniLM-L6-v2`, `algorithm='none'`, `target_dims=0`, `padding_method='noise_only'`),
holding everything else constant. This investigation analyses that data.

## Research questions

- **RQ1 (primary):** Does the number of chunk slots a document is padded/truncated to
  (`target_chunk_count`) causally affect downstream topic-model performance
  (NPMI, WEPS, WECS, NISH)? In which direction, and how large is the effect?
- **RQ2:** Is the effect family-specific (BERTopic, CombinedTM, GMM, KeyNMF, S³,
  ZeroShotTM) or corpus-specific?
- **RQ3:** Does the effect interact with the number of topics (10–200)?
- **RQ4:** How does runtime (doc-embedding, word-embedding, topic-model training)
  scale with chunk count?
- **RQ5 (practical):** Is there a performance/cost trade-off — can a smaller chunk
  count be used to save compute without losing topic quality?

## Data scope (pinned — every script must use exactly this)

- `pipeline.boe_cse_topic_model_performance` joined to
  `pipeline.boe_cse_topic_model_corpus_result` (+ `boe_cse_topic_model` for family
  names), `soft_delete = FALSE`. **6,000 metric rows / 1,500 results, fully complete.**
- Swept levels: imdb_reviews {2,4}, patent-classification {1,2},
  t2-ragbench-convfinqa {2,4,6}, wikipedia_sample {3,6,9}.
- 6 families × 5 num_topics {10,20,50,100,200} × 5 repeats per
  (corpus, chunk_count) cell.
- Timing: `pipeline.boe_cse_timing_result` (stages `boe_doc_embedding` 10,
  `boe_word_embedding` 10, `boe_topic_model` 1,500 rows).
- Cross-check only: main BOE d0/noise_only slice
  (`boe_topic_model_performance` ⋈ `boe_topic_model_corpus_result`,
  `soft_delete=FALSE AND target_dims=0 AND padding_method='noise_only'`) restricted to
  the four CSE corpora.
- ISH is negated to **NISH** at load time, so higher = better for all four metrics.

## Analyses

1. **A1 — Data overview & completeness:** design grid, row counts, verification that
   the cube is complete (it is; state it with evidence).
2. **A2 — Chunk-count effect on performance (primary):** per corpus × metric,
   summary stats per chunk level; paired Wilcoxon signed-rank between chunk levels on
   matched (family, num_topics) instance means (30 pairs/corpus); mean deltas as
   effect size; per-family breakdown with per-family Wilcoxon (5 pairs each, plus
   sign consistency); Spearman trend tests on the 3-level corpora; classification of
   each (family, corpus) cell as increases / decreases / no effect.
3. **A3 — Interaction with num_topics:** does the chunk-count effect depend on
   num_topics? Delta-vs-num_topics tables/plots per metric.
4. **A4 — Runtime scaling:** topic-model training time vs chunk count per family
   (means over repeats and num_topics, plus per-num_topics view); embedding-stage
   timings; approximate scaling statements.
5. **A5 — Performance/cost trade-off:** lowest vs highest swept chunk count per
   corpus: metric deltas alongside relative training-time change; practical
   recommendation.
6. **A6 — Consistency cross-check:** CSE cells at each corpus's natural chunk count
   vs the main-experiment d0/noise_only means (replication check).

## Statistical conventions (match investigations 1 & 2)

- Wilcoxon signed-rank for paired chunk-level comparisons (pairing unit =
  (family, num_topics) instance mean over 5 repeats), α = 0.05.
- Mann-Whitney U (p ≥ 0.05 ⇒ same group) for best/second-best grouping in per-corpus
  tables (\textbf{best group}, \underline{second group}).
- Spearman rank correlation for monotonic trends where ≥3 chunk levels exist.
- No multiple-testing correction (documented limitation, consistent with inv 1 & 2);
  report exact p-values so readers can apply their own correction.
- Mean ± std in summary tables; deltas reported in raw metric units.

## Deliverables

`ignore/investigation/` contains the analysis scripts (shared loaders in
`cse_common.py`), generated tables (`images/*.tex`) and figures (`images/*.png`),
and `report.tex` → `report.pdf` (pdflatex, run twice).

I want you to conduct a performance complementarity investigation for BOE topic models. This is Investigation 2, building on a prior Investigation 1.

## Context

### Prior investigation
The prior investigation is at:
`/home/darren/Documents/PhD/tm_framework/src/python/pipelines/boe_performance_comparison_investigation_1/`

Its key scripts are in `ignore/investigation/` (investigate.py, head_to_head.py, chunk_count_analysis.py, per_dataset_tables.py, etc.) and its findings are documented in `ignore/investigation/README.md` and `ignore/investigation/report.tex`. You should read these to understand the methodology, code patterns, and output format. You may reuse and adapt scripts from investigation_1.

### What changed since Investigation 1
- **New models**: KeyNMF and SemanticSignalSeparation are now available as BOE models (investigation_1 only had BERTopic, CombinedTM, GMM, ZeroShotTM)
- **Padding method resolved**: knn_mean and noise_only were shown to produce no notable difference. All new experiments use noise_only only. Filter to `padding_method = 'noise_only'` throughout.
- **New variable: target_dims**: Experiments now exist for both target_dims=50 and target_dims=100. Investigation 1 only had target_dims=100.
- **More repeats**: New experiments have 5 repeats per configuration (investigation_1 had 3).
- **Terminology**: Refer to non-BOE models as "existing" models (not "traditional").
- **Confirmed findings from investigation_1**:
  - Performance complementarity exists between BOE models and existing models
  - Performance complementarity exists among BOE models themselves

### Database access
- Run `. .env` then `psql $DB_URI <query>`
- BOE data: `pipeline.boe_topic_model_performance` joined with `pipeline.boe_topic_model_corpus_result` and `pipeline.boe_topic_model`
- Existing model data: `pipeline.topic_model_performance` joined with `pipeline.topic_model_corpus_result`, `pipeline.topic_model`, and `pipeline.corpus`
- Filter BOE data to `soft_delete = FALSE` and `padding_method = 'noise_only'`
- Metrics: NPMI, WEPS, WECS, ISH (invert ISH to NISH so higher = better for all metrics)

### Data dimensions (noise_only only)
- 6 BOE models x 2 target_dims = 12 BOE algorithm variants
- 10 corpora, 5 num_topics values (10, 20, 50, 100, 200), 5 repeats, 4 metrics
- 11 existing models with 10 repeats each

## Output location
Place all outputs in:
`/home/darren/Documents/PhD/tm_framework/src/python/pipelines/boe_performance_comparison_investigation_2/ignore/investigation/`

Create a `plan.md` and update it as you go. Generate images in an `images/` subfolder. Write a `README.md` documenting findings and a `report.tex` LaTeX report.

## Analyses to conduct

### Analysis 1: Target Dims Comparison (50 vs 100)
This is the primary new analysis. For each of the 6 BOE models, compare target_dims=50 vs target_dims=100:
- **Performance summary table**: Mean +/- std per (model, target_dims, metric), with significance markers (Mann-Whitney U)
- **Pairwise heatmaps**: Treat each (model, target_dims) combo as a separate algorithm (12 total). Generate win-count heatmaps across 50 problem instances (10 corpora x 5 num_topics).
- **Per-model target_dims sensitivity**: For each model, is there a statistically significant difference between dims=50 and dims=100? On which metrics? Use Wilcoxon signed-rank on matched (corpus, num_topics) pairs.
- **Conclusion per model and overall**: Classify as "complementary" (each dims wins on different metrics/conditions), "no difference", or "one dominates".

### Analysis 2: Expanded BOE Complementarity
Replicate the complementarity analysis from investigation_1 but with all 12 BOE variants (6 models x 2 target_dims):
- Performance summary table (Table 1 equivalent)
- Pairwise comparison heatmaps (Figure 1 equivalent)
- Pareto frontier analysis (Figure 2 / Table 2 equivalent)
- Per-dataset tables with significance markers

### Analysis 3: Updated BOE vs Existing Models
Combine all 12 BOE variants with the 11 existing models (23 algorithms total):
- Combined performance summary table
- Combined pairwise heatmaps
- Combined Pareto analysis
- Focus: Do the new BOE models (KeyNMF_boe, S3_boe) add value beyond what was shown in investigation_1?

### Analysis 4: Head-to-Head by Target Dims
For models that have existing counterparts (BERTopic, CombinedTM, ZeroShotTM, GMM), compare:
- BOE_dims50 vs existing (sbert only)
- BOE_dims100 vs existing (sbert only)
- BOE_dims50 vs existing (openai only)
- BOE_dims100 vs existing (openai only)
- Does one target_dims setting consistently close the gap (or widen the advantage) vs existing models?

## Technical notes
- Use `uv` for dependency management
- Use subagents for parallelisable work
- Name BOE variants as `{Model}_d{dims}` (e.g., `BERTopic_d50`, `KeyNMF_d100`)
- Reuse data-loading and plotting utilities from investigation_1 where possible
- Pareto metric pairs: (NPMI, WEPS), (NPMI, WECS), (WEPS, NISH), (WECS, NISH)
- Problem instance = (corpus, num_topics) -> 50 instances total

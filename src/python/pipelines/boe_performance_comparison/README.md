# BOE Performance Comparison

This pipeline compares BOE and standard topic-model performance metrics and generates:

- legacy BOE-vs-standard plots
- output-style algorithm heatmaps and Pareto plots for BOE/openai/sbert variants
- per-family BOE/openai/sbert comparison plots
- a rank/performance summary LaTeX table

It does not write any data to the database.

## Inputs

- `pipeline.boe_topic_model_performance`
- `pipeline.boe_topic_model_corpus_result`
- `pipeline.boe_topic_model`
- `pipeline.topic_model_performance`
- `pipeline.topic_model_corpus_result`
- `pipeline.topic_model`
- `pipeline.corpus`

## Matching

### Legacy alignment (BOE vs standard)

Rows are aligned by:

- `corpus_name`
- `model_group`
- `num_topics`
- `metric_name`

Model grouping rules:

- `BERTopic` and `BERTopic_sbert` are grouped as `BERTopic`
- `ZeroShotTM` and `ZeroShotTM_sbert` are grouped as `ZeroShotTM`
- all other model names are exact-match

### Variant alignment (family plots and output-style comparisons)

Variant names are derived as:

- BOE source rows: `<family>_boe`
- standard rows with base model name: `<family>_openai`
- standard rows with `_sbert` suffix: `<family>_sbert`

Default families:

- `BERTopic`
- `CombinedTM`
- `ZeroShotTM`

## Metric direction rule

- `ISH` is transformed to `NISH = -ISH` so higher is better.

## Output

Default output directory:

- `src/python/pipelines/boe_performance_comparison/ignore/plots`

Generated subdirectories:

- `legacy/`
- `algorithm_heatmaps/`
- `pareto/`
- `family_comparisons/`
- `tables/`

### Legacy outputs (`legacy/`)

Per metric:

- `dist_<metric>.png`
- `delta_scatter_<metric>.png`
- `delta_heatmap_<metric>.png`

### Algorithm heatmaps (`algorithm_heatmaps/`)

- `algorithm_heatmaps_variants.png`
- `algorithm_heatmap_<metric>_variants.png`

### Pareto plots (`pareto/`)

- `pareto_plots_variants.png`
- `pareto_plot_<metric1>_vs_<metric2>_variants.png`

### Family comparisons (`family_comparisons/`)

For each default family and metric:

- `family_<family>_dist_<metric>.png`
- `family_<family>_delta_matrix_<metric>.png`

### Tables (`tables/`)

- `performance_summary_variants.tex`

## Missing-data behavior

The pipeline skips gaps and uses available aligned data. It logs missing variants and missing metric coverage, and only skips individual outputs when insufficient data exists for those outputs.

## Run

```bash
PYTHONPATH=src/python python src/python/pipelines/boe_performance_comparison/main.py
```

Optional flags:

```bash
PYTHONPATH=src/python python src/python/pipelines/boe_performance_comparison/main.py \
  --output-dir src/python/pipelines/boe_performance_comparison/ignore/plots \
  --metrics NPMI,WEPS,WECS,ISH \
  --families BERTopic,CombinedTM,ZeroShotTM \
  --min-pairs 1 \
  --generate-output-style-plots true \
  --generate-family-variant-comparisons true \
  --generate-rank-table true
```

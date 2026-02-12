# BOE Performance Comparison

This pipeline compares BOE and standard topic-model performance metrics and generates plots only.
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

Rows are aligned by:

- `corpus_name`
- `model_group`
- `num_topics`
- `metric_name`

Model grouping rules:

- `BERTopic` and `BERTopic_sbert` are grouped as `BERTopic`
- `ZeroShotTM` and `ZeroShotTM_sbert` are grouped as `ZeroShotTM`
- all other model names are exact-match

Metric direction rule:

- `ISH` is transformed to `NISH = -ISH` so higher is better.

## Output

Default output directory:

- `src/python/pipelines/boe_performance_comparison/ignore/plots`

Per metric, three files are generated:

- `dist_<metric>.png`
- `delta_scatter_<metric>.png`
- `delta_heatmap_<metric>.png`

## Run

```bash
PYTHONPATH=src/python python src/python/pipelines/boe_performance_comparison/main.py
```

Optional flags:

```bash
PYTHONPATH=src/python python src/python/pipelines/boe_performance_comparison/main.py \
  --output-dir src/python/pipelines/boe_performance_comparison/ignore/plots \
  --metrics NPMI,WEPS,WECS,ISH \
  --min-pairs 1
```

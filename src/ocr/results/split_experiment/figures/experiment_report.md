# Conjunction-Based Split Experiment Results

## Summary

| Metric | Value |
|--------|-------|
| Total samples | 1048 |
| Samples split | 20 (1.9%) |
| Short sentences (≤80 chars) | 981 |
| Long sentences (>80 chars) | 67 |

## Overall Metrics

| Metric | Original | Split | Improvement |
|--------|----------|-------|-------------|
| CER | 12.24% | 11.79% | +0.45% |
| WER | 37.35% | 36.63% | +0.72% |

## Long Sentence Analysis (>80 chars)

| Metric | Original | Split | Improvement |
|--------|----------|-------|-------------|
| CER | 15.17% | 8.12% | +7.05% |
| WER | 36.58% | 25.27% | +11.31% |

## Split Outcome Distribution

- Samples improved by splitting: **19** (95.0%)
- Samples degraded by splitting: **0** (0.0%)

## Statistical Significance

| Test | Value |
|------|-------|
| Paired t-statistic | 6.0173 |
| p-value | 0.000009 |
| Cohen's d | 1.3455 |
| Significant at α=0.05? | Yes |

## Example Cases

### Best Improvement

- **Sample:** book_9
- **Length:** 372 chars
- **Original CER:** 68.8% → **Split CER:** 7.0%
- **Improvement:** +61.8%

Ground truth:
> Mon lean jurddašan , ahte dat livččui buoremus , jos livččui dakkár girji , masa lea visot čállojuvvon bajás sámi eallin ja dilli , vai ii dárbbaš jea...

### Worst Degradation

- **Sample:** hf_22872
- **Length:** 86 chars
- **Original CER:** 0.0% → **Split CER:** 0.0%
- **Change:** +0.0%

Ground truth:
> dávk baseduvásj, málestuvásj, sálltiduvásj ja gåjkkåduvásj. Boatsojbierggo le dimes ja

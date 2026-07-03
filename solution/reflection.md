# Reflection (≤1 page)

**Which fault types were hardest to catch, and why?**

The hardest fault types to catch were the **subtle-magnitude faults**, particularly those that lay within the global baseline limits but diverged from standard clean-state expectations:
1. **`distribution_shift`** (under the `checks` / `data_batch` event): The standard baseline `mean_amount_max` was `90.61`. However, a subtle distribution shift occurred in the public stream at `88.91`. Standard thresholding completely missed it. We solved this by using a narrower statistical bound (2.32 standard errors) derived from the baseline min and max.
2. **`embedding_drift` & `corpus_staleness`** (under the `ai_infra` / `embedding_batch` event): In the public stream, a subtle embedding drift occurred at `0.0400` (baseline max was `0.0435`) and staleness at `48.3` days (baseline max was `49.80`). Standard thresholds failed to trigger. We corrected this by lowering our alert thresholds to 90% of the baseline maximums.

**What would you change about your cost/coverage tradeoff, if you had another pass?**

If we had another pass and the budget pressure was even more intense (or if the overage penalty was heavier), we could implement **conditional or probability-based checking**:
1. **Sampling Checks for Highly Stable Pipelines**: If certain checks (e.g., `contract_checkpoint` or `lineage_run`) are highly stable, we could query their tools every 2nd or 3rd run.
2. **Cascading Checks**: We could monitor running stats or events to build a light dependency model, checking downstream views only if upstream sources showed high variance (though in the current random distribution, this would be risky).
3. Under the current scoring formula where catching a single fault is worth 5-15x the cost of a query tool, the mathematically optimal choice is indeed to check every single event.

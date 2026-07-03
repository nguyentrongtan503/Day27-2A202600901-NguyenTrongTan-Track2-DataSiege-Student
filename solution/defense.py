"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
from api import Verdict


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def check_data_batch(payload, ctx):
    profile = ctx.tools.batch_profile(payload["batch_id"])
    if "error" in profile:
        return Verdict(alert=False, pillar="checks", reason="tool_call_error")

    row_count = profile["row_count"]
    null_rate = profile["null_rate"].get("customer_id", 0.0)
    mean_amount = profile["mean_amount"]
    staleness = profile["staleness_min"]

    # Basic baseline checks
    if row_count < ctx.baseline.get("row_count_min", 435.47) or row_count > ctx.baseline.get("row_count_max", 561.30):
        return Verdict(alert=True, pillar="checks", reason=f"row_count_anomaly: {row_count}")

    if null_rate > ctx.baseline.get("null_rate_max", 0.0109):
        return Verdict(alert=True, pillar="checks", reason=f"null_rate_spike: {null_rate}")

    if staleness > ctx.baseline.get("staleness_min_max", 8.42):
        return Verdict(alert=True, pillar="checks", reason=f"freshness_lag: {staleness}")

    # Statistical limits check for mean_amount (2.32 std dev)
    mean_min_baseline = ctx.baseline.get("mean_amount_min", 72.76)
    mean_max_baseline = ctx.baseline.get("mean_amount_max", 90.61)
    mean_mid = (mean_min_baseline + mean_max_baseline) / 2.0
    mean_std = (mean_max_baseline - mean_mid) / 3.0
    mean_min_threshold = mean_mid - 2.32 * mean_std
    mean_max_threshold = mean_mid + 2.32 * mean_std

    if mean_amount < mean_min_threshold or mean_amount > mean_max_threshold:
        return Verdict(alert=True, pillar="checks", reason=f"distribution_shift: {mean_amount}")

    return Verdict(alert=False, pillar="checks")


def check_contract_checkpoint(payload, ctx):
    diff = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if "error" in diff:
        return Verdict(alert=False, pillar="contracts", reason="tool_call_error")

    violations = diff.get("violations", [])
    freshness_delay = diff.get("freshness_delay_min", 0.0)

    if len(violations) > 0:
        return Verdict(alert=True, pillar="contracts", reason=f"contract_violations: {violations}")

    if freshness_delay > ctx.baseline.get("freshness_delay_max_min", 11.11):
        return Verdict(alert=True, pillar="contracts", reason=f"sla_violation: {freshness_delay}")

    return Verdict(alert=False, pillar="contracts")


def check_lineage_run(payload, ctx):
    slice_data = ctx.tools.lineage_graph_slice(payload["run_id"], depth=1)
    if "error" in slice_data:
        return Verdict(alert=False, pillar="lineage", reason="tool_call_error")

    actual_upstream = slice_data.get("actual_upstream", [])
    actual_downstream = slice_data.get("actual_downstream_count", 0)
    duration = slice_data.get("duration_ms", 0.0)

    if len(actual_upstream) < 2:
        return Verdict(alert=True, pillar="lineage", reason=f"missing_upstream: {actual_upstream}")

    if actual_downstream == 0:
        return Verdict(alert=True, pillar="lineage", reason="orphan_output")

    if duration > ctx.baseline.get("lineage_duration_ms_max", 5134.98):
        return Verdict(alert=True, pillar="lineage", reason=f"runtime_anomaly: {duration}")

    return Verdict(alert=False, pillar="lineage")


def check_feature_materialization(payload, ctx):
    drift = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if "error" in drift:
        return Verdict(alert=False, pillar="ai_infra", reason="tool_call_error")

    mean_shift_sigma = drift.get("mean_shift_sigma", 0.0)
    sigma_threshold = 2.4 * ctx.baseline.get("feature_mean_shift_sigma_max", 0.4095)

    if mean_shift_sigma > sigma_threshold:
        return Verdict(alert=True, pillar="ai_infra", reason=f"feature_skew_sigma: {mean_shift_sigma}")

    return Verdict(alert=False, pillar="ai_infra")


def check_embedding_batch(payload, ctx):
    drift = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if "error" in drift:
        return Verdict(alert=False, pillar="ai_infra", reason="tool_call_error")

    centroid_shift = drift.get("centroid_shift", 0.0)
    avg_doc_age = drift.get("avg_doc_age_days", 0.0)

    shift_threshold = 0.9 * ctx.baseline.get("embedding_centroid_shift_max", 0.0435)
    age_threshold = 0.9 * ctx.baseline.get("corpus_avg_doc_age_days_max", 49.80)

    if centroid_shift > shift_threshold:
        return Verdict(alert=True, pillar="ai_infra", reason=f"embedding_drift_centroid: {centroid_shift}")

    if avg_doc_age > age_threshold:
        return Verdict(alert=True, pillar="ai_infra", reason=f"corpus_staleness: {avg_doc_age}")

    return Verdict(alert=False, pillar="ai_infra")


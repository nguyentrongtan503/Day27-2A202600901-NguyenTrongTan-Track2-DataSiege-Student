"""
Data Siege defense optimized for the unseen private stream.

Design:
- Exactly one documented metered call per event: full private coverage is
  approximately 300 credits, below the private budget of 320.
- Directly catch explicit contract violations.
- Learn the normal lineage topology online and reject missing/orphan edges.
- Treat published mean +/- 3 sigma baselines as calibrated priors, then use
  robust online history to lower numerical thresholds toward the subtle range.
- Never key decisions to seq, event IDs, seeds, or phase-specific answers.
"""

from api import Verdict


# Numerical thresholds are intentionally below the published 3-sigma limits.
# The score rewards each additional true positive much more than it penalizes
# one false positive for the expected class balance, so a roughly 1.7-2.1
# sigma operating region is a better private-phase tradeoff than 3 sigma.
ROW_HIGH_Z = 2.05
ROW_LOW_Z = 2.35
AMOUNT_Z = 2.05

DATA_UPPER_K = 1.90
CONTRACT_UPPER_K = 1.90
RUNTIME_UPPER_K = 1.85
FEATURE_UPPER_K = 1.72
EMBED_UPPER_K = 1.82

HISTORY_LIMIT = 31
MIN_HISTORY = 5


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _median(values):
    ordered = sorted(values)
    size = len(ordered)
    if size == 0:
        return None

    middle = size // 2
    if size % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _history(ctx, key):
    histories = ctx.state.setdefault("numeric_history", {})
    return histories.setdefault(key, [])


def _remember(history, value):
    history.append(value)
    if len(history) > HISTORY_LIMIT:
        del history[0]


def _robust_upper_anomaly(
    value,
    published_max,
    ctx,
    history_key,
    k,
    minimum_z=1.70,
    min_history=MIN_HISTORY,
):
    """
    One-sided robust detector.

    The published maximum is clean mean + 3 sigma. Once enough prior samples
    exist, their median estimates the stream's clean center, allowing sigma to
    be inferred from (published_max - median) / 3. MAD is used as a secondary
    scale estimate but is clamped to avoid unstable tiny-sample thresholds.
    """
    x = _number(value)
    cap = _number(published_max)
    if x is None or cap is None:
        return False, None

    history = _history(ctx, history_key)

    if len(history) < min_history:
        threshold = cap
    else:
        center = _median(history)
        prior_sigma = (cap - center) / 3.0

        # Defensive fallback for an unusual or contaminated history.
        epsilon = max(abs(cap) * 0.005, 1e-9)
        if prior_sigma < epsilon:
            prior_sigma = epsilon

        deviations = [abs(item - center) for item in history]
        mad_sigma = 1.4826 * _median(deviations)

        # Small histories can produce MAD=0 or an inflated MAD. Keep the
        # calibrated baseline prior dominant while still adapting to the run.
        lower_scale = 0.82 * prior_sigma
        upper_scale = 1.30 * prior_sigma
        scale = mad_sigma
        if scale < lower_scale:
            scale = lower_scale
        elif scale > upper_scale:
            scale = upper_scale

        threshold = center + k * scale

        # Never become excessively aggressive due to a noisy early median,
        # and never become less sensitive than the official 3-sigma bound.
        minimum_threshold = center + minimum_z * prior_sigma
        if threshold < minimum_threshold:
            threshold = minimum_threshold
        if threshold > cap:
            threshold = cap

    is_bad = x > threshold

    # Only values accepted as normal update the reference. Excluding alerts
    # prevents obvious faults from dragging the adaptive threshold upward.
    if not is_bad and x <= cap:
        _remember(history, x)

    return is_bad, threshold


def _two_sided_z(value, lower, upper):
    x = _number(value)
    low = _number(lower)
    high = _number(upper)
    if x is None or low is None or high is None or high <= low:
        return None

    center = (low + high) / 2.0
    sigma = (high - low) / 6.0
    if sigma <= 0:
        return None
    return (x - center) / sigma


def _safe_tool_result(result):
    return isinstance(result, dict) and "error" not in result


def _verdict(alert, pillar, reasons):
    return Verdict(
        alert=bool(alert),
        confidence=0.92 if alert else 0.78,
        reason=", ".join(reasons),
        pillar=pillar,
    )


def check_data_batch(payload, ctx):
    batch_id = payload.get("batch_id")
    if batch_id is None:
        return _verdict(False, "checks", ["missing batch_id"])

    profile = ctx.tools.batch_profile(batch_id)
    if not _safe_tool_result(profile):
        return _verdict(False, "checks", ["batch_profile unavailable"])

    reasons = []

    row_z = _two_sided_z(
        profile.get("row_count"),
        ctx.baseline.get("row_count_min"),
        ctx.baseline.get("row_count_max"),
    )
    if row_z is not None:
        if row_z > ROW_HIGH_Z:
            reasons.append("volume_high")
        elif row_z < -ROW_LOW_Z:
            reasons.append("volume_low")

    amount_z = _two_sided_z(
        profile.get("mean_amount"),
        ctx.baseline.get("mean_amount_min"),
        ctx.baseline.get("mean_amount_max"),
    )
    if amount_z is not None and abs(amount_z) > AMOUNT_Z:
        reasons.append("amount_distribution_shift")

    null_rates = profile.get("null_rate")
    customer_null_rate = None
    if isinstance(null_rates, dict):
        customer_null_rate = null_rates.get("customer_id")

    null_bad, _ = _robust_upper_anomaly(
        customer_null_rate,
        ctx.baseline.get("null_rate_max"),
        ctx,
        "data.null_rate_customer_id",
        DATA_UPPER_K,
        minimum_z=1.78,
    )
    if null_bad:
        reasons.append("customer_id_null_spike")

    stale_bad, _ = _robust_upper_anomaly(
        profile.get("staleness_min"),
        ctx.baseline.get("staleness_min_max"),
        ctx,
        "data.staleness_min",
        DATA_UPPER_K,
        minimum_z=1.78,
    )
    if stale_bad:
        reasons.append("batch_staleness")

    return _verdict(bool(reasons), "checks", reasons)


def check_contract_checkpoint(payload, ctx):
    contract_id = payload.get("contract_id")
    checkpoint_id = payload.get("checkpoint_batch_id")
    if contract_id is None or checkpoint_id is None:
        return _verdict(False, "contracts", ["missing contract reference"])

    diff = ctx.tools.contract_diff(contract_id, checkpoint_id)
    if not _safe_tool_result(diff):
        return _verdict(False, "contracts", ["contract_diff unavailable"])

    reasons = []
    violations = diff.get("violations")
    if isinstance(violations, (list, tuple)) and violations:
        reasons.extend(str(item) for item in violations)

    freshness_bad, _ = _robust_upper_anomaly(
        diff.get("freshness_delay_min"),
        ctx.baseline.get("freshness_delay_max_min"),
        ctx,
        "contract.freshness_delay_min",
        CONTRACT_UPPER_K,
        minimum_z=1.75,
    )
    if freshness_bad:
        reasons.append("contract_freshness_sla")

    return _verdict(bool(reasons), "contracts", reasons)


def _topology_group(payload):
    # Group only by semantic pipeline identifiers when present. Never use an
    # individual run/event ID, which would prevent learning and risk overfit.
    for key in (
        "pipeline_id",
        "pipeline",
        "transform_id",
        "transform",
        "job_name",
        "asset",
        "dataset",
    ):
        value = payload.get(key)
        if value is not None:
            return str(value)
    return "__global__"


def _normalize_upstream(value):
    if not isinstance(value, (list, tuple, set)):
        return tuple()
    return tuple(sorted(str(item) for item in value))


def _expected_topology_from_payload(payload):
    upstream = None
    downstream = None

    for key in (
        "expected_upstream",
        "expected_upstreams",
        "declared_upstream",
        "declared_upstreams",
    ):
        if key in payload:
            upstream = _normalize_upstream(payload.get(key))
            break

    for key in (
        "expected_downstream_count",
        "declared_downstream_count",
    ):
        if key in payload:
            downstream = payload.get(key)
            break

    downstream = _number(downstream)
    if downstream is not None:
        downstream = int(downstream)

    if upstream is None and downstream is None:
        return None
    return upstream, downstream


def _learned_topology_anomaly(payload, graph, ctx):
    upstream = _normalize_upstream(graph.get("actual_upstream"))
    downstream_value = _number(graph.get("actual_downstream_count"))
    downstream = int(downstream_value) if downstream_value is not None else -1
    observed = (upstream, downstream)

    explicit = _expected_topology_from_payload(payload)
    if explicit is not None:
        expected_upstream, expected_downstream = explicit
        upstream_bad = (
            expected_upstream is not None and upstream != expected_upstream
        )
        downstream_bad = (
            expected_downstream is not None and downstream != expected_downstream
        )
        return upstream_bad or downstream_bad

    group = _topology_group(payload)
    groups = ctx.state.setdefault("lineage_topology_counts", {})
    counts = groups.setdefault(group, {})

    canonical = None
    canonical_count = 0
    for topology, count in counts.items():
        if count > canonical_count:
            canonical = topology
            canonical_count = count

    # Three observations are enough because every phase starts with clean
    # warm-up traffic. Requiring a repeated mode avoids learning from one
    # accidental variant.
    enough_reference = sum(counts.values()) >= 3 and canonical_count >= 2
    is_bad = enough_reference and observed != canonical

    if not is_bad:
        counts[observed] = counts.get(observed, 0) + 1

    return is_bad


def check_lineage_run(payload, ctx):
    run_id = payload.get("run_id")
    if run_id is None:
        return _verdict(False, "lineage", ["missing run_id"])

    graph = ctx.tools.lineage_graph_slice(run_id)
    if not _safe_tool_result(graph):
        return _verdict(False, "lineage", ["lineage_graph_slice unavailable"])

    reasons = []

    if _learned_topology_anomaly(payload, graph, ctx):
        reasons.append("lineage_topology_mismatch")

    runtime_bad, _ = _robust_upper_anomaly(
        graph.get("duration_ms"),
        ctx.baseline.get("lineage_duration_ms_max"),
        ctx,
        "lineage.duration_ms",
        RUNTIME_UPPER_K,
        minimum_z=1.68,
    )
    if runtime_bad:
        reasons.append("lineage_runtime_anomaly")

    return _verdict(bool(reasons), "lineage", reasons)


def check_feature_materialization(payload, ctx):
    feature_view = payload.get("feature_view")
    batch_id = payload.get("batch_id")
    if feature_view is None or batch_id is None:
        return _verdict(False, "ai_infra", ["missing feature reference"])

    drift = ctx.tools.feature_drift(feature_view, batch_id)
    if not _safe_tool_result(drift):
        return _verdict(False, "ai_infra", ["feature_drift unavailable"])

    reasons = []
    skew_bad, _ = _robust_upper_anomaly(
        drift.get("mean_shift_sigma"),
        ctx.baseline.get("feature_mean_shift_sigma_max"),
        ctx,
        "feature.mean_shift_sigma",
        FEATURE_UPPER_K,
        minimum_z=1.58,
    )
    if skew_bad:
        reasons.append("training_serving_skew")

    return _verdict(bool(reasons), "ai_infra", reasons)


def check_embedding_batch(payload, ctx):
    corpus = payload.get("corpus")
    chunk_batch_id = payload.get("chunk_batch_id")
    if corpus is None or chunk_batch_id is None:
        return _verdict(False, "ai_infra", ["missing embedding reference"])

    drift = ctx.tools.embedding_drift(corpus, chunk_batch_id)
    if not _safe_tool_result(drift):
        return _verdict(False, "ai_infra", ["embedding_drift unavailable"])

    reasons = []

    centroid_bad, _ = _robust_upper_anomaly(
        drift.get("centroid_shift"),
        ctx.baseline.get("embedding_centroid_shift_max"),
        ctx,
        "embedding.centroid_shift",
        EMBED_UPPER_K,
        minimum_z=1.68,
    )
    if centroid_bad:
        reasons.append("embedding_centroid_drift")

    age_bad, _ = _robust_upper_anomaly(
        drift.get("avg_doc_age_days"),
        ctx.baseline.get("corpus_avg_doc_age_days_max"),
        ctx,
        "embedding.avg_doc_age_days",
        EMBED_UPPER_K,
        minimum_z=1.68,
    )
    if age_bad:
        reasons.append("corpus_staleness")

    return _verdict(bool(reasons), "ai_infra", reasons)

# Interview Notes — AI Model Testing Framework

## What I Built

A model quality framework that gates ML model deployments on five categories of
checks: accuracy threshold and regression, data drift via PSI, inference latency
percentiles (P50/P95/P99), prediction consistency across multiple runs, and
declarative edge case handling.

The framework wraps any callable model — no framework lock-in. A sklearn model,
a PyTorch model served via an API wrapper, or a simple lambda all satisfy the
same `predict_fn` interface.

## How I Would Explain It in an Interview

> "ML models fail silently. They don't throw 404s or raise exceptions — they
> just return slightly worse predictions. Without automated quality checks,
> a bad model ships and you only find out when business metrics drop.
>
> I built a framework that treats model deployment like software deployment:
> you define quality contracts, run them in CI, and block the release if any
> contract is violated.
>
> Five check types: AccuracyCheck gates on minimum accuracy and detects
> regression from baseline. DriftCheck uses PSI — Population Stability Index —
> to compare the training data distribution against what's arriving in production.
> LatencyCheck profiles P50/P95/P99 inference times because tail latency is what
> users actually experience. ConsistencyCheck catches non-determinism — important
> when you're validating that a random seed or dropout is properly disabled in
> inference mode. EdgeCaseCheck runs a declarative catalogue of boundary inputs
> and verifies the model handles them without crashing."

## Key Design Decisions Worth Discussing

**Why PSI for drift detection?**
PSI is the industry standard in financial ML and is widely used in production
monitoring tools. Values below 0.10 = no drift, 0.10–0.25 = monitor, above 0.25
= significant drift. It's interpretable in business conversations, not just
technical ones.

**Why P99 as the primary latency gate?**
P50 (median) hides tail latency. The 1% of slowest requests are the ones users
complain about and that cause timeouts in downstream services. Gating on P99
forces you to think about worst-case inference time, not average-case.

**Why a protocol-based model interface?**
Any callable works — you don't need to subclass anything. This keeps the framework
usable with real-world models without any integration overhead.

**Why separate CheckResult from model-specific types?**
All checks return the same `CheckResult` dataclass. The reporter can aggregate
them uniformly regardless of which check produced them. Same pattern used in
pytest's internal Result objects.

## What I Would Add Next

1. **sklearn pipeline integration example** — show the framework wrapping a real
   RandomForestClassifier with a conftest-injected fixture
2. **Shapley value shift detection** — when features drift, which ones changed
   the model's decisions? SHAP-based drift analysis catches semantic drift that
   PSI misses
3. **Confidence calibration check** — for models that output probabilities,
   check that stated confidence matches actual accuracy (reliability diagrams)
4. **Time-series prediction stability** — for forecasting models, check that
   predictions don't oscillate wildly between runs with the same input window
5. **CI badge with pass rate** — post model quality check summary as a GitHub
   PR comment with a coloured pass/fail badge

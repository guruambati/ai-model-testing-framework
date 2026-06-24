# Resume Bullets — AI Model Testing Framework

## Option A — AI QA Engineer / AI SDET focus

- Built a protocol-agnostic ML model quality framework (Python, pytest) with five
  check categories — accuracy threshold/regression, PSI-based data drift, P99
  latency profiling, prediction consistency, and declarative edge case handling —
  with 62 tests and GitHub Actions CI integration

## Option B — LLM Evaluation / MLOps QA focus

- Designed and implemented a model deployment quality gate covering Population
  Stability Index drift detection, P50/P95/P99 latency percentile gates, and
  A/B model comparison; framework is callable-agnostic and works with any predict
  function including sklearn, PyTorch, and LLM API wrappers

## Option C — QA Automation / Test Engineering focus

- Developed a reusable CheckResult-based reporting layer that aggregates model
  quality check outputs into console summaries and JSON reports; CI workflow blocks
  deployment on any ERROR-severity check failure using assert_all_pass()

## Notes on Usage

- Strong talking point: "PSI is the same metric used in financial services ML
  monitoring — it's interpretable in business terms, not just technical ones.
  Below 0.10 means no significant drift, above 0.25 means stop and investigate."
- For interviews that ask about LLM evaluation: "The same framework applies to
  LLM endpoints. You wrap the API call in a predict_fn lambda, and all five
  checks work identically — accuracy against golden labels, drift in prompt
  embedding space, P99 latency against your SLA."

# Dialogue QA benchmark working rules

- Keep the core flow simple: runnable Python baseline, natural dialogue,
  graph QA and external QA, grouped business requirements, then fixed
  acceptance and paired evaluation.
- Do the offline contract and recovery checks before a real run. During a run,
  preserve candidates, warnings, failures, receipts, and partial usage instead
  of stopping the whole collection for one local item.
- Retry bounded transient failures and resume from the latest valid checkpoint.
  Continue independent QA groups, requirements, and cases. Stop globally only
  for authentication failure, corrupted inputs, unavailable host, or exhausted
  total budget.
- Keep internal IDs, control fields, private plans, and audit-only answers out
  of model-facing dialogue, QA, requirements, and memory text.
- Graph QA measures historical coverage. External QA measures information that
  the final repository cannot reliably recover. Group related external QA into
  natural business requirements; do not turn every QA into a small bug task.
- Static fields such as type, difficulty, source closure, counts, and status
  should be computed by code whenever possible. Model prompts and outputs stay
  short and natural.
- Never claim a dataset is complete when it only has provisional, failed, or
  below-target artifacts. Keep those artifacts for review and report the
  shortfall.

# Dialogue QA Benchmark

This repository builds useful, evidence-grounded questions from a dialogue.
It is independent of any memory product: it does not connect to a memory
service and it does not scan the current repository to fill missing facts.

## What it does

The builder treats a dialogue as a time-ordered evidence stream:

1. Normalize visible messages, tool records, code observations, patches,
   failures, and test results.
2. Replay supplied patches in memory and build a lightweight evidence graph.
   Nodes represent dialogue/code objects and historical versions; edges
   represent explicit references, changes, failures, feedback, tests, and
   ordering.
3. Select small, related subgraphs. Distant records can be combined when they
   refer to the same file, function, decision, failure, or requirement.
4. Extract source-linked facts from small chunks, then generate and review QA
   from only the selected evidence.
5. Keep candidate, review, and publication results separately so rejected or
   unfinished questions remain inspectable locally.

The graph is a reconstruction of supplied dialogue, not a replacement for a
repository checkout. A question is valuable when its answer depends on a past
change, feedback, failure, test, decision, or multi-stage conversation that a
current checkout alone cannot reveal.

Two tracks are available:

- General QA: `single-hop`, `multi-hop`, `temporal`, `open-domain`, and
  `adversarial`, following the LoCoMo-style definitions adapted to dialogue
  stages.
- Code QA: `fact_recall`, `history_tracking`, `behavior_inference`, and
  `failure_diagnosis`. Code questions must rely on history, feedback,
  failures/tests, or a cross-location relationship; a current signature or
  isolated constant is not enough.

Difficulty is computed locally from the selected evidence structure: number of
necessary stages/versions, graph distance, and whether the question crosses a
failure/change/verification chain. The model does not choose the difficulty.

## Install and run

Python 3.9+ is required. From this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests -v
```

Static inspection (no network request):

```sh
python -m dialogue_benchmark.cli examples/dialogue.json \
  --output runs/example --qa-mode both
```

To generate with a chat model, provide an HTTPS endpoint and model explicitly,
then opt in to sending the selected evidence:

```sh
export BENCHMARK_API_KEY='...'
python -m dialogue_benchmark.cli examples/dialogue.json \
  --output runs/llm-example \
  --qa-mode both \
  --allow-network \
  --endpoint 'https://example.invalid/v1/chat/completions' \
  --model 'your-model'
```

The input file may be a unified JSON document or a supported native rollout.
The supplied records can contain visible messages, paired tool calls/results,
code observations, and successful patches. Paths are treated as evidence only;
the builder never reads those paths from disk.

Each run is written to its own directory. It contains normalized evidence,
scopes, facts, candidate questions, audit information, and the public QA view.
Run directories are private and ignored by Git. Open `viewer/index.html` and
use `viewer/build_data.py` to inspect a saved run locally.

## Design boundaries

- No repository scan or hidden context is used to answer a question.
- No API key, private credential, or raw private run is committed.
- Model calls receive a small evidence projection, not the entire dialogue.
- A failed chunk or review does not erase successful independent results.
- Automatic approval is a filter, not a human gold label; audit artifacts keep
  the evidence and rejection reason for manual inspection.

## Layout

`dialogue_benchmark/` contains normalization, graph construction, subgraph
selection, model interaction, quality checks, and the CLI. `tests/` contains
the regression suite. `examples/` contains synthetic, non-sensitive input.
The viewer is a local presentation of saved artifacts and is not part of the
evidence or memory system.

# Dialogue QA Benchmark

This repository builds useful, evidence-grounded questions from a dialogue.
The QA builder is independent of any memory product: it does not connect to a
memory service or scan a repository to fill missing historical facts. An optional
task experiment extends approved QA into new repository requirements.

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

General and code QA use the same six memory-purpose types, with independent
quotas. General questions concern recorded decisions and observations; code
questions connect that history to implementation or testing behavior.

| Type | Historical knowledge | Derived development task |
|---|---|---|
| `constraint_followthrough` | An agreed constraint and its scope | Extend or refactor while preserving that constraint |
| `correction_update` | An earlier rule and its later scoped correction | Apply the revised rule to the affected cases |
| `external_state_application` | A recorded user-side or environment fact | Adapt behavior to the observed conditions |
| `failure_avoidance` | A failed approach and its trigger | Implement new behavior without repeating that failure |
| `verification_reuse` | An actual test or experiment result | Use that result to choose boundaries and regression checks |
| `compatibility_preservation` | Established behavior required by existing callers | Add behavior while preserving the required old contract |

Each request has one statically nominated purpose. Focus and QA generation use
only its definition; a separate short review checks target alignment. Evidence,
atomicity, and completeness remain separate checks. The task author receives the
same purpose and must connect the actual historical answer to a new implementation
decision and observable acceptance behavior. Older QA types are not accepted; regenerate QA
before deriving tasks from an older run.

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
OpenHands public `session.jsonl` exports preserve paired tools and successful
file edits. A `dialogue.json` list of user/assistant messages is also accepted;
that list alone does not contain the tool history. Native multiline tool output
keeps its line boundaries, and complete edit bodies are represented once in the
version evidence instead of repeated inside tool metadata.
The supplied records can contain visible messages, paired tool calls/results,
code observations, and successful patches. Paths are treated as evidence only;
the builder never reads those paths from disk.

Each run is written to its own directory. It contains normalized evidence,
scopes, facts, candidate questions, audit information, and the public QA view.
Run directories are private and ignored by Git. Open `viewer/index.html` and
use `viewer/build_data.py` to inspect a saved run locally.

## Repository task pilot

Convert once with `python convert_session.py /path/to/session.jsonl --output runs/episode/input`.
Use the resulting `dialogue.json` as QA input. Native OpenHands tool results,
file changes, and original record locations survive the round trip;
`conversion.json` records input hashes and counts. Prefer the native session log
over a message-only export when extracting code-history questions.

This separate command reuses the agent-session simulator's OpenHands SDK runtime
and offline Docker environments. Run it with that simulator's Python environment,
configured images, and provider credentials. The selected QA run must include
approved questions and their saved generation requests.
Task agents do not inherit a per-response output token cap from the simulator
checkpoint. Provider limits and the task's total token and request budgets apply.

```sh
python -m dialogue_benchmark.task_eval.run \
  --simulator-path /path/to/agent-session-simulator \
  --source-run /path/to/completed-session \
  --qa-run runs/qa-with-inputs \
  --env-file /path/to/provider.env \
  --output runs/task-pilot \
  --count 3 --task-budget 6 --revisions 5 --workers 2
```

The configured model receives a historical QA, its actual generation input, and
read-only access to the dialogue-end repository. It authors a new requirement,
tests where feasible, and acceptance criteria.
An independent Code Agent implements the requirement; a validator checks the
baseline, reference implementation, and test quality. Failed attempts remain
available for inspection, with up to five revisions by default.
`count` targets completed task pairs. Distinct QA-derived requirements are tried
in bounded batches until the target, QA pool, or `task-budget` is exhausted.
The default task budget is twice the target; failure records are retained.

The dialogue-end code is pinned as an independent local Git baseline, with no
upstream remote. `--baseline` can reuse an already pinned clean repository.
Agents receive code copies without Git metadata or other solutions. Each
reference and trial retains its full code, a binary-capable Git patch, and a
`version.json` receipt tying it to the baseline commit and result tree. Patch
application is checked in a disposable clone, including deletions and file modes.
To restore a result, clone the baseline, check out the receipt's base commit,
then run `git apply --index /path/to/changes.patch` in that clone.

A requirement must solve a real new problem, with a historical rule that changes
observable behavior. Main historical tasks use still-applicable external rules;
recoverable information is excluded and uncertain availability needs review.
Before trials, the validator checks the exact injected answer against all active
rules, including scoped corrections and exceptions. Missing answer content returns
for QA correction or a smaller task; it is never filled from private acceptance notes.

The frozen acceptance table has four columns: ID, requirement, basis, and check.
Each mandatory row cites the new task or a historical rule and an exact test name,
a saved shell check, or a repeatable inspection. The program retains individual
JUnit results and executes the same frozen tests/regressions with the Code Agent's
project path configuration. It also replays a saved wrong implementation which
retains the new functionality but violates a historical rule. This must fail the
historical check while passing the basic functionality checks.

Both fresh solvers start from the same pinned code. Only the memory condition
receives the QA answer. Both can explicitly request history; the responder answers
only that question from the same frozen public history, then resumes the session.
Normal completion does not invoke the responder. Asking history is counted as an
interaction, not a task failure.

Test-linked acceptance results are computed directly. The Judge checks only
predeclared inspection items with resolvable file/line evidence. Historical
compliance is derived from these same acceptance rows, not a second subjective
judgment. A demonstrated violation fails; every mandatory item satisfied passes;
remaining missing evidence is uncertain. Test-complete results do not depend on a
Judge response. New counterexamples are retained for a shared check revision,
rather than silently changing one trial's criteria.

New runs have no exploration checkpoint extraction, matching, recovery, or route
score. Reports retain every pair's correctness, per-rule outcomes, history requests,
development calls, reads/searches and tokens. Development calls exclude think and
finish; actual failed tool calls still count. `paired-differences.json` contains
with-minus-without differences; cost deltas are computed only when both pass.
Old run files remain readable without creating missing evidence.

Metrics include tool calls,
explicit file views, separately counted shell read/search observations, provider
tokens, and available reasoning-text length. Script-based reads are not fully
counted. File
and reasoning token estimates are labeled separately from provider token usage;
unavailable reasoning is not treated as zero. Elapsed time is not a comparison
metric. This pilot measures supplied historical-answer context, not retrieval by
a memory system.

For a combined run laid out as `input/`, `baseline/`, `qa/`, and `tasks/`, use
`python render_run.py runs/episode` to package the existing QA viewer and one
HTML entry point. The task report updates after each completed task.

`run_episode.py --source-run /path/to/completed-session --simulator-path
/path/to/agent-session-simulator --env-file /path/to/provider.env --output
runs/episode` runs all stages in order using the simulator's configured model.
Defaults target 40 approved questions per track and 12 evaluated tasks, exploring
up to 24 requirements. QA workers default to 10; repository workers default to 3.
Inputs are extracted once per unique source across overlapping subgraphs;
static code relations are reused for QA grouping rather than repeated in fact requests.
For a larger input, initial relationship proposals use shared-object lookups and
bounded samples spanning the timeline. Expansion choices are computed only when
a fact enters a selected group; the underlying version and code relations remain
available. `--reuse-facts /path/to/previous-qa-run` resumes after extraction without
model calls for facts. It requires exactly the same normalized records and chunk
layout, and preserves previous extraction failures. Reused request usage is not
counted as newly consumed tokens.
After QA completes, `--resume-tasks` reuses that QA and the pinned baseline,
checking the input hash before starting repository work. Move any previous
`tasks/` directory aside first to preserve its attempts. Reference files are
exported as sandbox-readable copies; private originals retain their permissions.
Completed agents export their original model responses and tool events to a verified
compressed trace, then remove their recorded containers, snapshot volumes and networks.
Finished tests also release their sandboxes. Images remain reusable. Interrupted agents
and timed-out tests retain their resumable state; compaction is deferred while any remain unfinished.
The full episode command automatically compacts completed runs. For an existing completed
episode, run `python compact_run.py runs/episode`. It keeps the pinned baseline, accepted
reference and both trial implementations, patches, frozen criteria/tests, acceptance
evidence, metrics, and compressed traces. Parsed QA candidates retain their generation inputs
and review reasons, including rejected candidates. Failed requirements keep their text,
criteria, test source, test output and failure reasons;
temporary code copies, repeated request histories, batch snapshots and diagnostic backups
are removed only after all executions finish and sandbox resources are released successfully.
Clarification requests share their historical context in one compressed audit file;
actual questions, replies and per-call usage remain inspectable. Optional design probes
retain replay-verified patches and traces instead of a full additional code copy.
HTML links to compressed traces instead of embedding another full copy. Shared evidence is stored once and
subgraphs use lossless references. New code copies omit mypy/ruff caches; existing frozen
versions retain their original contents and hashes.
Each agent also has a 20-minute execution ceiling to bound runaway tool commands;
this is a resource limit, not an evaluation metric.

## Design boundaries

- QA extraction uses only input evidence; repository access belongs to the
  optional task-construction and execution command.
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

## Public episode packages and historical constraints

`run_episode.py --episode-manifest /path/to/package/manifest.json --simulator-path
/path/to/agent-session-simulator --env-file /path/to/provider.env --output runs/episode`
accepts `memory-episode-v1` packages. The manifest identifies the public dialogue,
cutoff event, final snapshot and independent control configuration. Paths stay
inside the package; dialogue and snapshot hashes are checked before generation.
Model configuration uses environment variable names, never inline keys.

The public `model-visible-dialogue-v1` stream contains messages, tool calls and
the actual tool text sent to the model. It does not reconstruct complete versions
from private editor old/new metadata. General QA includes public tool evidence in
the surrounding discussion stage. `--source-event EVENT_ID` or
`--source-object path/to/file.py::symbol` can be repeated to select public fact
roots in either track. Other facts remain available for related expansion;
selecting a root does not make nearby facts causal evidence.

For these packages, task construction freezes source-backed historical statements,
their applicable scopes and explicit replacement links in `history.json`. A later
rule overrides only its stated scope. The independent validator checks the public
sources and updates, the exact QA answer supplied in the oracle condition, and
the new task's observable acceptance behavior. New tasks must not redo already
completed construction work. The informed reference proves feasibility; answer sufficiency is separately
checked before freezing. `--design-probe` optionally
runs an independent no-memory construction probe; its success is not an admission
requirement. The probe does not establish a scored route or alter task admission.

The two scored conditions remain no memory and oracle QA answers. Both can ask
the same read-only responder for frozen historical information, continuing the
same Code conversation. It answers only an actual question supported by the
frozen sources; it cannot inspect candidate code or invent current environment
facts. Responder calls share the total request/token budget with execution, and
their usage and delivered replies are retained separately. There is no additional
per-response output cap. Current environment checks use the existing code tools.

Reports separate historical application, correctness and costs. They distinguish
cross-session re-asks, repeated questions after an answer in the same session,
and reasonable confirmation of a potentially changed condition. An incomplete
review remains insufficient evidence; an observed compliant result does not by
itself prove memory caused it. No real memory retriever is evaluated. Existing
raw-session runs remain readable under their original input contract.

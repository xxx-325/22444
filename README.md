# Dialogue QA Benchmark

This repository builds useful, evidence-grounded questions from a dialogue.
The QA builder is independent of any memory product: it does not connect to a
memory service or scan a repository to fill missing historical facts. An optional
read-only final-repository probe can remove questions that the current checkout
already answers directly. An optional task experiment extends approved QA into
new repository requirements.

## What it does

The external-information workflow is: public historical events → QA → one related
development requirement per qualifying QA → frozen acceptance → paired execution
with and without the historical answer. It uses one QA pool and M1–M6 memory types.
External QA workflow selection, focus, generation, repair, and evidence review share
later public User/Code messages up to the cutoff for scoped corrections, even across
tasks without a declared revision link. Fact extraction uses only the event's
declared source IDs; the shared context does not add fact seeds or unrelated tool logs.

The optional graph mode treats a dialogue as a time-ordered evidence stream:

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

In graph mode, general and code QA use the same six memory-purpose types, with independent
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

By default, generation does not fix one purpose in advance. The model first
chooses one useful follow-up target from the selected evidence and writes the
question; deterministic evidence rules then assign the applicable type. Passing
`--general-types` or `--code-types` explicitly constrains that choice. A separate
short review checks target alignment, while evidence, atomicity, and completeness
remain separate checks. The task author receives the final type and must connect the actual historical answer to a new implementation
decision and observable acceptance behavior. Older QA types are not accepted; regenerate QA
before deriving tasks from an older run.

Graph-mode difficulty is computed locally from the selected evidence structure: number of
necessary stages/versions, graph distance, and whether the question crosses a
failure/change/verification chain. The model does not choose the difficulty.

## Install and run

For concrete QA, an authored task draft, observed failures, and construction
costs, see the [real task-construction examples](examples/task-construction-review.md).

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

When a final repository snapshot is available, add `--repository /path/to/checkout`.
After QA review and before quotas are applied, a host-controlled probe may look up
and read relevant files. It receives the question, answer claims to verify, and
code anchors, without historical sources. Only repository content supports a
definitive recoverability decision. External-only publication holds uncertain
cases for review. Probe receipts are kept under
`recoverability/`, and the audit marks filtered candidates as
`filtered_recoverable`, not as quality rejections.

Each probe reply has exactly four required tags: `PROBE`, `REASON`, `QUERY`, and
`EVIDENCE`, with no end marker. Incomplete or conflicting replies fail closed;
evidence references must identify earlier repository queries. Both `recoverable`
and `history_required` decisions require cited file content from a read or content
search. Citing only empty results or filename-only matches leaves the result
`uncertain`.

### External-only QA source

External QA uses one target, `--qa-count`, and one exploration limit,
`--group-budget` (default: four times the target, at least ten). The target counts
only approved, safe, unique published questions. Each event group generates at
most one QA; a qualifying QA supplies one candidate development requirement.
The existing one-repair limit, review, deduplication, and repository probe remain.

| Memory type | What the answer preserves | Derived requirement |
|---|---|---|
| M1 · Interaction agreements | User rules and their scope | Apply the agreement to a new feature |
| M2 · External facts | External system contracts or business facts | Adapt observable behavior to those facts |
| M3 · Misleading repository information | Established corrections to stale code, docs, or examples | Implement according to the corrected behavior |
| M4 · Costly trial and error | Conclusions from actual failed attempts or expensive investigation | Handle the same conditions without repeating the problem |
| M5 · Runtime differences | Observed environment-dependent behavior | Support the relevant execution conditions |
| M6 · Cross-session state | Still-valid prior decisions or state, including scoped corrections | Continue work from the applicable state |

The producer supplies the type. Separate model calls choose a future business
workflow, identify the historical rules it needs, and write one QA. The extracted
external facts define the question target; surrounding material supplies context
and corrections to those same rules. Each call returns short tagged text.
The model writes only the direction, question, and
source-linked answer. A useful answer changes a future implementation choice,
behavior, boundary, or validation decision. A one-run byte total or file inventory
alone does not qualify. A recorded receiver size limit can qualify because it
changes how a future export must work. The existing target review checks this
distinction. External QA has no graph-distance difficulty label.

User-supplied API specifications alone are not external business facts. Extraction
keeps actual customer choices, authorizations, agreements, and external state with
their recorded objects, cycles, and conditions. A hypothetical later approval is
not an observed update; cycle-limited authorizations remain eligible when they
apply to the task. Future workflows add capabilities within that confirmed scope;
QA retrieves recorded decisions rather than seeking a new approval or confirmation.
Scoped corrections apply only when recorded in the supplied history.
One selection rule includes its scope, restrictions, and
exceptions; independent outputs or actions remain separate answer points.
A complete set or mapping that defines one rule stays in one answer point so its
membership boundary remains intact.
Parsed focus responses are saved before validation in local stage artifacts
(`focus-response.json`, and `focus-refinement-response.json` when applicable).

The dialogue producer may also save a small `external-events.json` sidecar. It
records public dialogue source IDs for facts that arose from a user correction,
an environment observation, a perturbation failure, a compatibility exception,
or a completed verification. Later public uses are optional evidence. Events
sharing an explicit `task_id`, or linked through `supersedes`, form one group;
`context_ids` supplies the public request and correction context. All linked
disclosures are collected before the group budget is applied. Private scenario
plans never supply the answer.

Run `--qa-source external --external-events /path/to/external-events.json` to
use only those events as QA seeds. This mode does not build the evidence graph
or search ordinary code facts. It extracts facts from the referenced dialogue,
generates and reviews QA, and still runs the optional final-repository probe so
questions whose complete answer is already recoverable are separated from the
external set. The default `--qa-source graph` path is unchanged.
Review uses the declared event and its supplied context; it does not require a
file or symbol anchor. Source closure and request-size checks still apply.
A candidate citing only retained context, such as a later correction, has static
evidence status `unknown` and proceeds to semantic review to establish its connection
to the selected rules. Context alone does not establish an external fact;
out-of-scope citations are rejected.
Questions name the receiving system and intended work while leaving the historical
limit or rule for the answer. The repository probe gets a final decision after its
last allowed read; that decision cannot issue another query.

`run_episode.py --episode-manifest ... --qa-source external` reads the event file
from the package and checks its byte hash. An explicit event file must match the
package. Each event requires a static `memory_kind` from M1–M6; this becomes the
primary QA type and is retained through requirement generation and reporting.
For a group with several disclosures, the latest supplies the primary type;
the scope retains every member's type. Related rules have separate answer points
within one workflow. The workflow and its original generation input accompany
the QA into task selection, which checks that this goal is feasible and still new
in the final repository instead of choosing an unrelated feature.
Evidence review accepts a user-confirmed agreement without requiring past
execution. Claims of application require a public tool result, including the outcome
of a matching command. Relevant corrections stay in the workflow, focus, QA, review
and task-history inputs. Repairs preserve the chosen workflow and their final source
references. Source and request-size checks apply throughout.
Target review checks the historical question; repository recoverability is checked
separately. Atomic-point repairs keep the question unchanged and reuse a passed
target judgment only when its inputs match. Revised answer points still undergo
relevance, atomicity, completeness, and evidence review.
A positive confirmation or application verdict must cite supplied material;
a missing citation is recorded as invalid review evidence, not a finding of non-use.
The workflow guides selection; questions remain short historical follow-ups rather
than development specifications or requests to recompute example outputs.
Each connected group enters the unified pool once, regardless of a producer's old track
label. General/code options belong to graph mode. Types are reported separately
without quotas that force all six types to appear.

```sh
python run_episode.py --episode-manifest /path/to/episode/manifest.json \
  --qa-source external --qa-count 20 --group-budget 80 \
  --task-count 10 --task-budget 20 \
  --simulator-path /path/to/agent-session-simulator \
  --env-file /path/to/provider.env --output runs/episode-memory
```

`qa-public.json` contains the unified final set. The local viewer filters final
questions and all candidates by memory type and links their historical evidence;
the task report retains each requirement's source QA and type.

### Generated project collections

`run_collection.py` composes the simulator's project preparation, scenario
preparation, progressive dialogue, export, and the existing episode evaluation.
Use [examples/collection.json](examples/collection.json) as a fixed attempt list.
Its `runtime_config` points to a simulator configuration containing the model,
immutable Docker images, and whole-project `max_requests` / `max_tokens` budgets.
Credentials remain in the separate environment file.

```sh
python run_collection.py --config /path/to/collection.json \
  --simulator-path /path/to/agent-session-simulator \
  --env-file /path/to/provider.env --output runs/project-pilot
```

Run with the simulator's Python environment, or supply it through `--python`.
New projects have a tested baseline and two or three consecutive feature commits.
A private development plan connects each increment to related outside information,
affected decisions and disclosure triggers. Scenario preparation reads this same plan;
the dialogue starts at the baseline and implements the increments itself.
The host checks frozen feature tests on the prior and new versions and preserves
previous regressions. An existing project can use `prepared_config` instead of a
business `brief`. Scenarios start independently from that project's base; each
paired trial starts from its dialogue's actual final snapshot.

The collection checks cumulative request/token usage between stages; a started
stage finishes under its own existing budgets. No per-response output cap is
introduced. Rejected stages remain recorded, and the runner does not add attempts
to replace failures. Use a new output directory for another fixed plan.
An episode with no exported external history is recorded as `no_external_history`
and skips QA/task execution. M4/M5 scenarios require prepared runtime conditions; design counts and public event
counts are reported separately. Root Git lineage groups project families.

For recall dataset construction, set `evaluation.qa_only` to `true`, or use
`run_episode.py --qa-only`. The run ends after QA review and repository recoverability
checks, saving the dialogue input, all candidates, source evidence and QA viewer.
It creates no repository tasks or paired trials. Both future evaluation conditions
use the same code and ordinary project documentation; historical knowledge documents
and QA answer keys stay outside their shared repository. Retrieved memory can be
supplied separately to the memory condition.

`collection.md` lists construction outcomes and costs; `report.md` and
`report.html` reuse the paired-trial report. `collection.json` retains the full
stage records. QA-only collections use `collection.md` and each episode's QA page
without a paired-trial report. Exact provider request bodies and responses are kept in private
verified compressed traces before disposable runtime files are removed.

The input file may be a unified JSON document or a supported native rollout.
OpenHands public `session.jsonl` exports preserve paired tools and successful
file edits. A `dialogue.json` list of user/assistant messages is also accepted;
that list alone does not contain the tool history. Native multiline tool output
keeps its line boundaries, and complete edit bodies are represented once in the
version evidence instead of repeated inside tool metadata.
The supplied records can contain visible messages, paired tool calls/results,
code observations, and successful patches. Paths are treated as evidence only by
the fact/QA stages; the optional final-repository probe is the only stage that
reads a supplied checkout.

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

Repeat an admitted task without generating or changing its requirement or checks:

```sh
python -m dialogue_benchmark.task_eval.repeat \
  --source-task runs/task-pilot/task-01 \
  --simulator-path /path/to/agent-session-simulator \
  --env-file /path/to/provider.env \
  --output runs/task-01-repeat
```

This runs exactly two fresh pairs: without memory then with memory, followed by
with memory then without memory. The source manifest supplies the model configuration
and total request, token and time budgets; no per-response output cap is added.
The runner verifies the original baseline and frozen specification hashes, copies
them into a new output, and retains the original task and oracle answer. Solvers
receive independent baseline copies; author and reference implementations are not
copied into their workspaces. Both groups keep the existing history clarification
mechanism. `source.json` records provenance, each `pair-NN/manifest.json` saves the
cumulative results, and the usual reports include failures and partial pairs.
Neither prior results nor successful repetitions select which pairs are retained.
Interrupted runs keep completed results and mark the remaining pair pending; use a
new output directory for another run. Invoke this command for each admitted task.

The host calls the configured model once per selection decision and executes one
structured read-only lookup/read request, retaining exact sources, ranges, and
pagination. Selection starts from the QA, repository observations, and an index
of historical sources. Original messages are read on demand so unrelated topics
in the same turn do not drive task selection. Full history stays on the host.
Duplicate requests, errors, and
exhausted budgets produce pending records, not ineligibility conclusions.
A candidate first freezes the historical targets from the original dialogue.
Then one tool-free call writes only the public task, and a second tool-free call
writes private history use and acceptance material. The existing qualification
review runs only after the public task is fixed; only then are tests constructed.
Selection, drafting, review, and test authoring share cumulative
budgets, without a per-response output cap.
The test author cannot change qualified requirements or historical rules.
Test repairs reuse the qualified draft. Compatibility tests apply to the named
old APIs; new APIs are checked by their stated behavior. Regression commands use
baseline tests copied into the frozen specification, so solver-authored tests do
not change the scored regression set.
The public draft preserves the selected project's goal and uses actual repository
observations and fixes a callable interface or command before tests are authored.
An unfinished test author stops with its own reason; changed draft
files are listed separately, without restarting construction automatically.
Use `--selection-only` to stop after qualification, retaining decisions, queries,
and usage without starting OpenHands or paired execution.
An independent Code Agent implements the requirement; a validator checks the
baseline, reference implementation, and test quality. Failed attempts remain
available for inspection, with up to five revisions by default.
One model request reviews the exact cited evidence, visible dialogue, frozen
rules and injected answer. Its decisions and quoted answer coverage are saved
under `history-review/`; unsupported history or incomplete answers stop admission.
The test author receives the fixed criteria and historical contract; source
archives stay with qualification and independent validation.
For small Python snapshots with a `tests/` directory, one model request receives
the source, tests, documentation and fixed criteria and writes the acceptance
tests. The host copies the original regression suite and executes all checks.
Snapshots exceeding the existing request budget use the OpenHands test author.
Reference implementations and both scored conditions continue to use OpenHands.
For historical tasks, a second finite review checks each acceptance row and any
additional tests against the task, contract, changed source files and executed
results. Missing coverage or unsupported requirements return for correction.
For tasks covered by executable checks, one model request supplies changed source
files for a historical-error variant. The host exports a replayable patch and
executes the frozen checks. Tasks with inspection items use OpenHands to produce
the variant and inspection evidence. These steps share a preflight budget; the complete
dialogue archive remains in the qualified draft and frozen task.
An interrupted validator retains its error and usage and stops construction;
it does not trigger a new task draft without a completed review.
Use `--reuse-preparation /path/to/task-01/construction-00 --count 1` in a new
task-run output to reuse a completed test author after a preflight interruption.
The runner verifies the QA, source evidence, baseline and qualified requirements,
then reruns qualification, reference implementation and validation. Original
artifacts remain unchanged and reused author usage is identified separately.
Add `--preparation-feedback /path/to/test-review.md` to repair those tests in one
model request with the full saved files, preserving the qualified task. The host
reruns the resulting tests and preflight before freezing anything.
This feedback should describe test defects, without solver comparison outcomes.
Collection and execution errors return to the test author before starting a
reference solver. Tests separate public functionality from historical rules;
compatibility assertions compare only the behavior required by the task.
`count` targets completed task pairs. Distinct QA-derived requirements are tried
in bounded batches until the target, QA pool, or `task-budget` is exhausted.
The default task budget is twice the target; failure records are retained.

`examples/controlled_report_fixture.py` creates a small report repository and a
scripted customer-protocol dialogue for testing this pipeline. It is a synthetic
mechanism fixture, not an automatically generated collaboration episode. Real
episodes use the same QA and task stages after dialogue generation.

`examples/batchsync_seed.py --repository runs/suite/seed --receipt runs/suite/seed.json`
creates a standard-library project with three adjacent reference commits for
projection, byte-bounded batches and receipt classification. Each revision runs
its offline tests before being committed. Feed these commits to the dialogue
simulator; the builder itself creates no messages or customer agreements.
For the full pipeline, pass `--qa-source external --external-events /path/to/external-events.json`
to `run_episode.py`. This also probes the pinned final repository for answer
recoverability before deriving tasks.
If no QA remains eligible, the pipeline saves an empty task report with
`no_eligible_qa` and finishes without starting development agents.

The dialogue-end code is pinned as an independent local Git baseline, with no
upstream remote. `--baseline` can reuse an already pinned clean repository.
Agents receive code copies without Git metadata or other solutions. Each
reference and trial retains its full code, a binary-capable Git patch, and a
`version.json` receipt tying it to the baseline commit and result tree. Patch
application is checked in a disposable clone, including deletions and file modes.
To restore a result, clone the baseline, check out the receipt's base commit,
then run `git apply --index /path/to/changes.patch` in that clone.
The run manifest also records evaluator and simulator commits, package content
hashes and execution budgets, so uncommitted source changes remain traceable.

A requirement must solve a real new problem, with a historical rule that changes
observable behavior. At least one required rule must need external history;
repository-recoverable compatibility requirements can also apply and are checked.
Uncertain availability needs review. Before trials, source review checks the exact
injected answer against necessary information still missing from the public task
and repository, including scoped corrections and exceptions. At least one such
gap must remain. Already supplied information need not be repeated in the answer.
Acceptance covers every active rule. Missing answer content returns
for QA correction or a smaller task; it is never filled from private acceptance notes.

The frozen acceptance table has four columns: ID, requirement, basis, and check.
Each mandatory row cites the new task or a historical rule and an exact test name,
a saved shell check, or a repeatable inspection. The program retains individual
JUnit results and executes the same frozen tests/regressions with the Code Agent's
project path configuration. It also replays a saved wrong implementation which
retains the new functionality but violates an external historical rule. This must fail the
historical check while passing the basic functionality checks.
Generated pytest tests use the program-provided `candidate_root` fixture; the
reserved `conftest.py` supplies the same repository path in authoring and all checks.
Static QA evidence marked `unknown` may proceed to semantic QA review; explicit
`insufficient` evidence is rejected. Neither QA approval nor static uncertainty
replaces task eligibility review.

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
Paired token differences count solver tokens; history-responder tokens are reported separately.
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
Episode QA extraction, generation, review, deduplication, repository probes, and
task-stage host model calls inherit `judge.request_timeout` from the episode's
control configuration or simulator checkpoint. Standalone QA accepts
`--request-timeout SECONDS` (default: 90); timeouts must be positive finite numbers.
An omitted host-call timeout retains the 90-second default.
`--model-request-chars` sets the serialized QA input limit (default: 32,000
characters). Increase it when a complete external event bundle needs more room
within the model's context window. Failed QA stages retain their evidence and
stop the episode as `qa_generation_failed`, separately from a completed run
with no eligible questions.
Collections default this limit to 96,000 characters and accept
`evaluation.model_request_chars` to configure it.
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

Run offline regressions with `python -m unittest discover -s tests -q`.
To include the simulator-to-QA workflow contract test, use the simulator's Python
environment and set `PYTHONPATH=/path/to/agent-session-simulator`. The test checks
planned steps, public disclosure and correction links, QA inputs and requirement
handoff using fixture model responses; it does not call a provider.

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
rule overrides only its stated scope. Source review checks the public sources,
updates and exact QA answer supplied in the oracle condition. The OpenHands
validator checks observable acceptance behavior. New tasks must not redo already
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

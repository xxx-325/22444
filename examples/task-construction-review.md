# Real QA-to-task construction examples

These excerpts come from the 2026-09-23 Click pilot. They show why two approved
QA did not produce an admitted development task. Source quotations retain their
original Chinese; explanations summarize the saved artifacts. Private paths,
credentials, full transcripts, and model reasoning are omitted.

The task-construction rerun used code revision `102d1fb`. QA were reused from
the preceding extraction run. The input contained two completed development
tasks, five user messages, five assistant messages, and 305 tool calls with
305 results. Extraction targeted one general QA and one code QA; it did not
exhaust all possible questions in that dialogue.

## Example 1: Correct information already available in the repository

Type: `compatibility_preservation`. QA status: `approved`.

**Question**

> 在 click 兼容层排查中，commands.md 记录的 Context Defaults 的 context_settings 覆盖行为，在新改动后必须对哪些调用方或条件继续成立？

**Answer points**

1. 通过命令装饰器传入 context_settings 来覆盖上下文默认值的行为，必须继续对使用该装饰器声明命令的调用方成立。
2. 该覆盖行为自 Click 2.0 起被记录为支持，因此对依赖该版本起行为的既有调用方必须继续成立。

The author returned `NO_TASK`: `src/click/core.py`, `docs/commands.md`, and
existing tests already describe or implement the behavior. This is a justified
exclusion from an experiment targeting information unavailable from the current
repository. It does not imply that the answer is false or that no other new task
could be proposed for Click.

The QA audit passed target alignment, relevance, atomicity, evidence, and
completeness. It also recorded `static_evidence_status=unknown`,
`static_evidence_reason=missing_type_evidence`, and `track=unknown`. Reconciling
that static uncertainty with final approval is an open implementation review
item. Approval alone does not establish suitability for an external-history task.

## Example 2: A narrow historical requirement became a broad new rule

Type: `constraint_followthrough`. QA status: `approved`.

**Historical user message, event 567**

> 你跑偏了：那个 fileno 的修复不是这次要做的，之前的问题已经收尾了。现在要的是把 docs/index.rst、docs/commands.rst、docs/changes.rst 转成 MyST Markdown，commands 页面到现在还是 reST 格式、也没有对应的 Markdown 页面。请按你之前列的计划把这三个页面转完，然后跑构建验证，确认能正常构建、内容呈现不变。

**Question**

> 按之前列出的计划把 docs/index.rst、docs/commands.rst、docs/changes.rst 转成 MyST Markdown 后，运行构建验证时要确认的约定标准是什么？

**Answer points**

1. 构建验证需要确认能正常构建。
2. 构建验证需要确认内容呈现不变。

The answer has direct user evidence. Its weakness for this experiment is that
it supplies only a generic validation goal, not a specific external behavior rule.

**Actual authored requirement excerpt**

> 新增一个文档页面 `docs/stream-lifecycle.md`，并在 `docs/index.md` 的 "General Reference" toctree 中登记它，使它能被构建并出现在文档导航里。

> 本次改动属于文档页面改动，沿用已确认的构建验证约定（见 `history-contract.txt` 的 h1）：改完后运行构建验证，必须确认文档能正常构建，且既有页面的内容呈现不变。

The draft requested sections on isolation, file descriptors, captured output,
and filesystem isolation. It added a second rule preserving root `CHANGES.rst`.

**Problems visible in the saved draft**

- The original scope was format conversion of three existing pages. The draft
  contract expanded it to adding or changing documentation generally.
- The public requirement repeated both injected answer points. Both execution
  arms would already receive that information from the task itself.
- The additional `CHANGES.rst` rule was marked active and repository-recoverable,
  but was absent from the injected answer.
- Tests checked selected headings and text in existing HTML pages. They did not
  establish full rendering equivalence or correctness of the new documentation.
- Contract citations used truncated UUIDs, while source validation requires the
  exact original event IDs.

**Observed tool sequence and stop**

1. Read QA, original evidence, historical message ranges, repository files, and
   documentation configuration.
2. Run two baseline Sphinx builds.
3. Begin writing artifacts at development tool call 29; save a task, historical
   contract, and acceptance test module.
4. Run the tests. They fail with `Cannot find source directory
   (/workspace/checks/docs)` because the test uses its own directory as the
   repository root.
5. Edit repository discovery. The edit references `os.environ` without importing
   `os`; it is not rerun before the total token budget is exhausted.

`acceptance.md` and `memory-use.md` were not saved. The recorded stopping reason
was budget exhaustion with incomplete author artifacts. The semantic problems
above were found by inspecting the draft afterward, not by a completed Validator.

## Recorded construction costs

| Candidate | Author outcome | Development tool calls | Total tokens |
|---|---|---:|---:|
| Context defaults compatibility | `NO_TASK` | 12 | 317,626 |
| Documentation build agreement | Budget exhausted; incomplete draft | 36 | 1,564,264 |
| Total | No admitted task | 48 | 1,881,890 |

For example 2, 34 provider requests used 1,548,417 input tokens and 15,847
completion tokens. These are cumulative request usage, not the length of a
single answer. Tool calls were 32 terminal calls and four file-editor calls.

Neither candidate reached a reference solution or paired with/without-memory
execution in this rerun. These numbers measure construction cost, not memory
benefit, solver success, or solver efficiency.

## Code review entry points

- `dialogue_benchmark/task_eval/prompts.py`: `AUTHOR`, `HISTORY_AUTHOR`,
  `HISTORY_VALIDATOR`, and `TASK_REVIEW`.
- `dialogue_benchmark/task_eval/run.py`: `construct()` checks the artifacts after
  the author conversation returns.
- `dialogue_benchmark/task_eval/history.py`: `prepare_history()`,
  `freeze_contract()`, and `oracle_coverage()`.
- `dialogue_benchmark/fact_index.py` and `dialogue_benchmark/quality.py`: static
  type evidence and QA approval behavior.

The main design questions are whether eligibility should be resolved before
writing tests, how to separate the external rule under study from ordinary
compatibility requirements, and how to preserve historical scope without
repeating the entire memory in the public task. The current
`freeze_contract()` requires every active rule to be external; this can reject
a mixed task even when its central rule genuinely needs history.

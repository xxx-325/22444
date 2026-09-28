"""Small shared contracts for the line-tagged QA protocol."""


MISSING_KINDS = {
    "earlier_state", "later_state", "reason", "outcome", "dependency",
}
QA_TYPE_GUIDANCE = {
    "constraint_followthrough": "Ask which previously agreed constraint applies to one concrete future action, preserving its object and conditions. An assistant suggestion alone is not an agreement.",
    "correction_update": "Ask how an earlier rule was corrected and where the revised rule applies. Require the old rule, the later correction, and its scope; a code edit alone is not a correction of a rule.",
    "external_state_application": "Ask for one recorded external rule needed for future work: a limit, mapping, exception, or observed result. State the system and intended work as context in the question; do not ask the reader to identify that system or repeat the work request. The answer supplies only the missing rule. Keep its actual values out of the question. Who supplied the rule is provenance, not another answer target. General knowledge and a guess from current code are not external observations.",
    "failure_avoidance": "Ask which previously attempted approach failed under which conditions and what that rules out for future work. Require an actual failure or explicit user feedback, not a hypothetical error branch.",
    "verification_reuse": "Ask what a recorded test or experiment established under specified conditions and which future check it informs. A planned test or a successful file edit is not a test result.",
    "compatibility_preservation": "Ask which historically established behavior must survive a new change, and for which callers or conditions. A version difference alone does not establish a compatibility requirement.",
}
QA_TYPES = frozenset(QA_TYPE_GUIDANCE)

# External episodes use their source labels; graph QA keeps its purpose taxonomy.
MEMORY_TYPE_LABELS = {
    "M1": "交互约定", "M2": "外部事实", "M3": "仓库误导",
    "M4": "高成本试错", "M5": "运行时差异", "M6": "跨会话状态",
}
MEMORY_TYPE_GUIDANCE = {
    "M1": "Ask for the previously agreed user rule needed by a concrete future feature. State the known customer and situation in the question, and preserve the rule's recorded scope in the answer.",
    "M2": "Ask which recorded fact about an external system or business context changes the behavior required of a future feature.",
    "M3": "Ask what previously established correction to misleading code, documentation, or examples a future change must account for.",
    "M4": "Ask what a completed trial established about a failed approach or costly investigation, under conditions relevant to future work.",
    "M5": "Ask which observed runtime or environment difference a future feature must handle, preserving the actual conditions.",
    "M6": "Ask which recorded, still-applicable decision or state later work must carry forward within its confirmed scope. Apply a scoped correction only if one is explicitly recorded.",
}
MEMORY_TYPES = frozenset(MEMORY_TYPE_GUIDANCE)
MEMORY_QA_RULES = """
The answer must determine behavior within one future business workflow.
Use that workflow only to identify a useful situation, not as a task specification
to paste into the question. Ask a short, natural follow-up about the missing history.
Several related historical rules may be necessary; keep each rule and its condition
in a separate answer point.
Ask for the missing historical decision, not a general inventory of requirements.
Only ask separately about applicability when the evidence contains distinct cases;
then each answer must explicitly pair a case with its rule. Do not append a generic
conditions question to a fixed situation. Express each rule once, without paraphrase points.
Every point must change a decision in that same workflow.
An isolated file inventory, byte total from one run, or test count is not such a decision.
Do not append a calculation of example outputs or counts to a question about a rule.
Use examples to identify its meaning or scope, not as extra recall targets.
A recorded external size limit can be useful when it determines how future output must behave.
Use the rule, condition, or established consequence actually present in the evidence;
do not turn a one-off observation into a permanent rule.
Preserve the publicly confirmed customer, object, cycle, and applicability.
A hypothetical later approval is not an actual update or scoped correction.
Do not extend a cycle-limited authorization to other cycles or call it the current policy.
"""

TASK_TYPE_GUIDANCE = {
    "constraint_followthrough": "让新功能或重构实际用到已确认约束；验收其适用条件下的行为。",
    "correction_update": "让新需求触及被纠正的规则；验收修订范围内使用新规则、范围外保留仍有效规则。",
    "external_state_application": "让新需求适配已记录的用户侧或环境事实；冻结观测条件，不把历史状态当成当前实测。",
    "failure_avoidance": "让新需求涉及过去失败的条件；验收功能正确及同类失败不再出现，不强制固定实现路线。",
    "verification_reuse": "让历史测试或实验结论影响新功能的边界或验证选择；新增测试必须验证新需求，不能只重跑旧测试。",
    "compatibility_preservation": "扩展或重构相关能力，验收新行为及已确认需要保留的旧调用行为。",
}
MEMORY_TASK_GUIDANCE = {
    "M1": "提出实际需要沿用用户既有约定的新功能；验收约定适用范围内的行为。",
    "M2": "提出依赖已记录外部事实的新功能；让该事实决定具体行为或边界。",
    "M3": "提出会触及已确认仓库误导的新功能；验收实际行为遵循历史纠正。",
    "M4": "提出涉及已有试错条件的新功能；用历史结论避免已证实的问题，不要求特定实现路线。",
    "M5": "提出需要适配已观察运行环境差异的新功能；在固定环境条件下验收。",
    "M6": "提出承接已确认状态或决定的新功能；只沿用仍有效的规则，保留局部纠正的范围。",
}
SIMPLE_ATOMICITY_STATES = {"single", "compound", "uncertain"}
SIMPLE_EVIDENCE_STATES = {"supported", "contradicted", "insufficient", "stale"}

SIMPLE_TEMPORAL_WORDING_RULE = (
    "A local evidence window does not prove that an event was the previous, latest, "
    "or final occurrence. In generated prose, do not write 上次, 上一次, 最近一次, "
    "or 最后一次. Use 之前 and identify the concrete test, error, task, or change. "
    "Preserve those words only when quoting supplied code or a literal string."
)

SIMPLE_ATOMICITY_RULE = (
    "One subject's old state plus new state is one transition claim. "
    "One condition plus its result is one behavior claim. "
    "One eligibility or selection rule, including the conditions, restrictions, and "
    "exceptions that define its scope, is one claim. 'Include all approved records "
    "in the same window, with no additional record-level exclusions' is single. "
    "A complete set or mapping defining one named rule is one claim, not a list "
    "of unrelated facts. 'The partner's allowed prefixes are exactly {AB, CD}' "
    "is single: splitting its members loses the complete-set boundary. "
    "Independent outputs or actions are separate claims, even when they share a "
    "condition. Use separate points "
    "for different subjects or independent conclusions. Examples: 'Port 8080 "
    "changed to 9090' is single. 'On timeout, do not publish output' is single. "
    "'The port changed and the retry count changed' is compound. 'tests contains "
    "a.py and b.py' is compound because each member can be true independently. "
    "'Expand each path to --config, so the CLI receives each path' is single. "
    "'Catch TimeoutExpired and raise Error with its message' is single. "
    "'The new timestamp is earlier than the frontier, so the comparison is false "
    "and the frontier error is not raised' is single. "
    "'Add a function parameter and pass it onward' is compound because the signature "
    "change and the value flow are independently true. "
    "Do not split a continuous range or path merely because it has multiple components."
)


def simple_point_ids(question):
    """Return the ordered local review IDs for one candidate."""
    return (["A%d" % (index + 1)
             for index in range(len(question.get("answer_points", [])))]
            + ["F%d" % (index + 1)
               for index in range(len(question.get("forbidden_points", [])))])


def required_point_ids_text(question):
    """Render the exact point checklist supplied to a focused reviewer."""
    return ",".join(simple_point_ids(question))

"""Small shared contracts for the line-tagged QA protocol."""


MISSING_KINDS = {
    "earlier_state", "later_state", "reason", "outcome", "dependency",
}
QA_TYPE_GUIDANCE = {
    "constraint_followthrough": "Ask which previously agreed constraint applies to one concrete future action, preserving its object and conditions. An assistant suggestion alone is not an agreement.",
    "correction_update": "Ask how an earlier rule was corrected and where the revised rule applies. Require the old rule, the later correction, and its scope; a code edit alone is not a correction of a rule.",
    "external_state_application": "Ask which recorded user-side or environment fact changes a future decision. Preserve when and where it was observed. General knowledge and a guess from current code are not external observations.",
    "failure_avoidance": "Ask which previously attempted approach failed under which conditions and what that rules out for future work. Require an actual failure or explicit user feedback, not a hypothetical error branch.",
    "verification_reuse": "Ask what a recorded test or experiment established under specified conditions and which future check it informs. A planned test or a successful file edit is not a test result.",
    "compatibility_preservation": "Ask which historically established behavior must survive a new change, and for which callers or conditions. A version difference alone does not establish a compatibility requirement.",
}
QA_TYPES = frozenset(QA_TYPE_GUIDANCE)

TASK_TYPE_GUIDANCE = {
    "constraint_followthrough": "让新功能或重构实际用到已确认约束；验收其适用条件下的行为。",
    "correction_update": "让新需求触及被纠正的规则；验收修订范围内使用新规则、范围外保留仍有效规则。",
    "external_state_application": "让新需求适配已记录的用户侧或环境事实；冻结观测条件，不把历史状态当成当前实测。",
    "failure_avoidance": "让新需求涉及过去失败的条件；验收功能正确及同类失败不再出现，不强制固定实现路线。",
    "verification_reuse": "让历史测试或实验结论影响新功能的边界或验证选择；新增测试必须验证新需求，不能只重跑旧测试。",
    "compatibility_preservation": "扩展或重构相关能力，验收新行为及已确认需要保留的旧调用行为。",
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
    "One condition plus its result is one behavior claim. Use separate points "
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

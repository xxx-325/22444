"""Small shared contracts for the line-tagged QA protocol."""


MISSING_KINDS = {
    "earlier_state", "later_state", "reason", "outcome", "dependency",
}
CODE_DISTINCTIVENESS_BASES = {"A", "B", "C", "D"}
SIMPLE_ATOMICITY_STATES = {"single", "compound", "uncertain"}
SIMPLE_EVIDENCE_STATES = {"supported", "contradicted", "insufficient", "stale"}

SIMPLE_TEMPORAL_WORDING_RULE = (
    "A local evidence window does not prove that an event was the previous, latest, "
    "or final occurrence. In generated prose, do not write 上次, 上一次, 最近一次, "
    "or 最后一次. Use 之前 and identify the concrete test, error, task, or change. "
    "Preserve those words only when quoting supplied code or a literal string."
)

BEHAVIOR_INFERENCE_CONTRACT = (
    "Behavior inference answers how one concrete condition, value, or dependency "
    "produces an observed behavior. The link may cross code locations, or may "
    "depend on an explicitly linked historical/version prerequisite that is "
    "necessary for the behavior. Merely recalling a constraint is fact recall; "
    "only comparing old and new states is history tracking; diagnosing an observed "
    "failure is failure diagnosis."
)

CODE_DISTINCTIVENESS_RULE = (
    "A code question must make at least one basis indispensable to its answer: "
    "A compares an earlier state with a later state; B uses a recorded failure, "
    "feedback, decision, or constraint; C connects conditions, calls, or data flow "
    "across different code locations, or connects an explicitly linked indispensable "
    "historical/version prerequisite to its implementation behavior. D is a single-location fact, surface inventory, "
    "or anything whose need for A, B, or C is not proven. Material count, source "
    "count, and historical wording do not prove a basis. A None value, path, or "
    "parameter is allowed only when its change, cause, or cross-location dependency "
    "is the actual answer target."
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

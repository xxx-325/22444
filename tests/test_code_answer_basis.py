"""Offline regressions for the code-only answer-basis review gate."""

import copy
import unittest

from dialogue_benchmark.llm import (
    CODE_DISTINCTIVENESS_PROMPT,
    ModelStageError,
    parse_text_response,
    review_candidates,
)
from tests.simple_test_helpers import maybe_relevance_response


class ScriptedClient:
    """Return fixed protocol documents while retaining safe request projections."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.usage = []

    def ask(self, prompt, payload):
        self.calls.append((prompt, copy.deepcopy(payload)))
        self.usage.append({"status": "completed"})
        relevance = maybe_relevance_response(prompt, payload)
        if relevance is not None:
            return relevance
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, str):
            return parse_text_response(response)
        return copy.deepcopy(response)


def distinctiveness_review(basis):
    return """REVIEW q1
review_contract: code_distinctiveness_v1
answer_basis: %s
END_REVIEW""" % basis


def atomicity_review(points="A1=single;F1=single"):
    return """REVIEW q1
review_contract: simple_atomicity_v1
point_atomicity: %s
END_REVIEW""" % points


def completeness_review():
    return """REVIEW q1
review_contract: simple_v1
completeness: complete
missing: none
END_REVIEW"""


def evidence_review(points="A1=supported@资料1;F1=contradicted@资料1"):
    return """REVIEW q1
review_contract: simple_v1
point_evidence: %s
END_REVIEW""" % points


class CodeAnswerBasisTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "cutoff": 8,
            "dialogue": [
                {"id": "e1", "kind": "message", "role": "assistant", "order": 1,
                 "stage_id": "s1", "text": "旧端口为 18080，旧 seed 为 0。"},
                {"id": "e2", "kind": "message", "role": "assistant", "order": 2,
                 "stage_id": "s1", "text": "后续修改将端口改为 18081，并将 seed 改为 None。"},
                {"id": "e3", "kind": "message", "role": "assistant", "order": 3,
                 "stage_id": "s1", "text": "旧方案在 timeout 后失败。"},
                {"id": "e4", "kind": "message", "role": "user", "order": 4,
                 "stage_id": "s1", "text": "反馈要求 timeout 最多重试两次并返回失败。"},
                {"id": "e5", "kind": "message", "role": "assistant", "order": 5,
                 "stage_id": "s1", "text": "runner 返回 timeout。"},
                {"id": "e6", "kind": "message", "role": "assistant", "order": 6,
                 "stage_id": "s1", "text": "adapter 将该 timeout 转换为失败结果。"},
                {"id": "e7", "kind": "message", "role": "assistant", "order": 7,
                 "stage_id": "s1", "text": "server.py 未提供 seed 时默认值为 None。"},
                {"id": "e8", "kind": "message", "role": "assistant", "order": 8,
                 "stage_id": "s1", "text": "tests 目录有 a.py、b.py、c.py。"},
            ],
            "events": [],
            "versions": [],
            "stages": [{"id": "s1"}],
            "review_guard_sources": [],
            "review_guard_complete": True,
            "model_request_chars": 60000,
            "max_context_chars": 60000,
        }
        self.facts = [
            {"id": "f1", "statement": "旧端口为 18080，后续改为 18081。",
             "sources": ["e1", "e2"]},
            {"id": "f2", "statement": "旧方案在 timeout 后失败，反馈要求最多重试两次。",
             "sources": ["e3", "e4"]},
            {"id": "f3", "statement": "runner 的 timeout 被 adapter 转换为失败结果。",
             "sources": ["e5", "e6"]},
            {"id": "f4", "statement": "未提供 seed 时默认值为 None。",
             "sources": ["e7"]},
            {"id": "f5", "statement": "tests 目录有 a.py、b.py、c.py。",
             "sources": ["e8"]},
        ]

    def candidate(self, question, answer, answer_sources, *, qa_type,
                  forbidden="相反的实现不会发生。", forbidden_sources=None,
                  difficulty="medium", track="inference_control"):
        return {
            "id": "code-q1",
            "candidate_id": "code-q1",
            "model_id": "q1",
            "qa_mode": "code",
            "type": qa_type,
            "category": qa_type,
            "track": track,
            "difficulty": difficulty,
            "question": question,
            "answer_points": [{"text": answer, "sources": list(answer_sources)}],
            "forbidden_points": [{
                "text": forbidden,
                "sources": list(forbidden_sources or answer_sources[:1]),
            }],
        }

    def run_code(self, candidate, basis, *, responses=None, allow_repair=True):
        scripted = responses or [
            distinctiveness_review(basis),
            atomicity_review(),
            completeness_review(),
            evidence_review(),
        ]
        client = ScriptedClient(scripted)
        result = review_candidates(
            self.scope, self.facts, [candidate], client,
            qa_mode="code", review_mode="simple", allow_repair=allow_repair)
        return result, client

    def test_a_b_c_continue_after_basis_and_preserve_annotations(self):
        cases = [
            (
                "A",
                self.candidate(
                    "端口的旧值和修改后的新值分别是什么？",
                    "端口从 18080 改为 18081。", ["e1", "e2"],
                    qa_type="history_tracking", difficulty="hard",
                    track="history_core"),
            ),
            (
                "B",
                self.candidate(
                    "旧方案失败后，反馈要求采取什么重试约束？",
                    "失败反馈要求 timeout 最多重试两次并返回失败。", ["e3", "e4"],
                    qa_type="failure_diagnosis", difficulty="medium"),
            ),
            (
                "C",
                self.candidate(
                    "当 runner 返回 timeout 时，adapter 如何处理？",
                    "runner 的 timeout 被 adapter 转换为失败结果。", ["e5", "e6"],
                    qa_type="behavior_inference", difficulty="hard"),
            ),
        ]
        for basis, candidate in cases:
            with self.subTest(basis=basis):
                original = copy.deepcopy(candidate)
                result, client = self.run_code(candidate, basis)
                self.assertEqual(result["questions"][0]["status"], "approved")
                self.assertEqual(
                    [receipt["stage"] for receipt in client.usage],
                    ["review_code_distinctiveness", "review_relevance",
                     "review_atomicity", "review_completeness",
                     "review_evidence"],
                )
                reviewed = result["questions"][0]
                self.assertEqual(reviewed["type"], original["type"])
                self.assertEqual(reviewed["category"], original["category"])
                self.assertEqual(reviewed["difficulty"], original["difficulty"])

                prompt, payload = client.calls[0]
                self.assertIn("code_distinctiveness_v1", prompt)
                self.assertIn("answer_basis: A|B|C|D", prompt)
                self.assertIn("A mainly compares", prompt)
                self.assertIn("B mainly asks", prompt)
                self.assertIn("C mainly derives", prompt)
                self.assertEqual(
                    set(payload), {"materials", "relations", "facts", "candidates"})
                self.assertNotIn("requested_task", payload)
                for key in ("scope", "dialogue", "events", "versions", "stages",
                            "cutoff", "difficulty", "type"):
                    self.assertNotIn(key, payload)
                self.assertEqual(
                    set(payload["candidates"][0]),
                    {"id", "immutable_question", "answer_points", "forbidden_points"},
                )

    def test_d_surface_or_historically_wrapped_current_value_rejects_before_repair(self):
        candidates = [
            self.candidate(
                "tests 目录中有哪些文件？", "tests 目录有 a.py、b.py、c.py。", ["e8"],
                qa_type="fact_recall", forbidden="tests 目录没有这些文件。"),
            self.candidate(
                "历史上 server.py 当前 seed 的默认值是什么？",
                "未提供 seed 时默认值为 None。", ["e7"],
                qa_type="history_tracking", forbidden="默认 seed 为 0。",
                track="history_core"),
        ]
        for candidate in candidates:
            with self.subTest(question=candidate["question"]):
                result, client = self.run_code(candidate, "D")
                self.assertFalse(result["questions"])
                self.assertEqual(result["rejected"][0]["reason"],
                                 "code_answer_not_distinctive")
                self.assertEqual(result["rejected"][0]["failed_checks"],
                                 ["code_answer_distinctive"])
                self.assertEqual(
                    [receipt["stage"] for receipt in client.usage],
                    ["review_code_distinctiveness"],
                )
                self.assertFalse(result.get("revisions"))

    def test_target_mismatch_can_rewrite_question_once_then_runs_full_review(self):
        """A type-target mismatch may fix the question, but keeps its identity."""
        candidate = self.candidate(
            "当 runner 返回 timeout 时，adapter 如何处理？",
            "runner 的 timeout 被 adapter 转换为失败结果。", ["e5", "e6"],
            qa_type="history_tracking", difficulty="hard", track="history_core")
        original_payload = {
            "facts": [
                {"text": "旧端口为 18080，后续改为 18081。",
                 "materials": ["资料1", "资料2"]},
                {"text": "runner 的 timeout 被 adapter 转换为失败结果。",
                 "materials": ["资料3", "资料4"]},
            ],
            "materials": [
                {"reference": "资料1", "text": "旧端口为 18080。"},
                {"reference": "资料2", "text": "后续端口改为 18081。"},
                {"reference": "资料3", "text": "runner 返回 timeout。"},
                {"reference": "资料4", "text": "adapter 转换为失败结果。"},
            ],
            "relations": ["资料1 precedes 资料2", "资料3 calls 资料4"],
            "focus": {"text": "比较端口的旧状态与后续状态",
                      "sources": ["资料1", "资料2"]},
        }
        context = {
            "prompt": "ORIGINAL QA PROMPT\nTARGET_DEFINITION: compare the port's earlier and later state.",
            "payload": copy.deepcopy(original_payload),
            "ref_to_source": {
                "资料1": "e1", "资料2": "e2", "资料3": "e5", "资料4": "e6",
            },
        }
        repaired_response = """QA q1
TYPE: behavior_inference
DIFFICULTY: easy
QUESTION: 端口的旧值和修改后的新值分别是什么？
ANSWER_POINT: 端口从 18080 改为 18081。 || SOURCES: 资料1,资料2
FORBIDDEN_POINT: 端口仍为 18080。 || SOURCES: 资料1
END_QA"""
        client = ScriptedClient([
            distinctiveness_review("C"),
            repaired_response,
            distinctiveness_review("A"),
            atomicity_review(),
            completeness_review(),
            evidence_review("A1=supported@资料1;F1=contradicted@资料1"),
        ])
        repair_state = {"remaining": 1}
        result = review_candidates(
            self.scope, self.facts, [candidate], client,
            qa_mode="code", review_mode="simple", allow_repair=True,
            generation_context=context, repair_state=repair_state)

        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual([receipt["stage"] for receipt in client.usage], [
            "review_code_distinctiveness", "repair",
            "review_code_distinctiveness", "review_relevance",
            "review_atomicity", "review_completeness", "review_evidence",
        ])
        self.assertEqual(sum(receipt["stage"] == "repair" for receipt in client.usage), 1)
        self.assertEqual(repair_state["remaining"], 0)

        revision = result["revisions"][0]
        self.assertEqual(revision["original_review"]["reason"],
                         "code_answer_basis_target_mismatch")
        self.assertEqual(revision["original_review"]["failed_checks"],
                         ["code_answer_target_aligned"])
        self.assertEqual(revision["before"]["id"], candidate["id"])
        self.assertEqual(revision["after"]["id"], candidate["id"])
        self.assertEqual(revision["after"]["type"], candidate["type"])
        self.assertEqual(revision["after"]["category"], candidate["category"])
        self.assertNotEqual(revision["after"]["question"], candidate["question"])
        self.assertEqual(result["questions"][0]["type"], candidate["type"])
        self.assertEqual(result["questions"][0]["category"], candidate["category"])

        repair_prompt, repair_payload = client.calls[1]
        self.assertTrue(repair_prompt.startswith(context["prompt"]))
        self.assertIn("For a task mismatch, restore the original fixed task and focus",
                      repair_prompt)
        for key, value in original_payload.items():
            self.assertEqual(repair_payload[key], value)
        self.assertEqual(
            repair_payload["original_candidate"]["answer_points"][0]["sources"],
            ["资料3", "资料4"])
        self.assertIn("different task", repair_payload["review_issue"])
        self.assertEqual(
            client.calls[2][1]["candidates"][0]["immutable_question"],
            revision["after"]["question"])

    def test_distinctiveness_format_or_http_failure_is_review_only(self):
        malformed = """REVIEW q1
review_contract: code_distinctiveness_v1
answer_basis: E
END_REVIEW"""
        candidate = self.candidate(
            "端口的旧值和修改后的新值分别是什么？",
            "端口从 18080 改为 18081。", ["e1", "e2"],
            qa_type="history_tracking", track="history_core")
        cases = [
            (malformed, "invalid_code_distinctiveness_review"),
            (ModelStageError("http_error", http_status=503),
             "review_code_distinctiveness_failed"),
        ]
        for response, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                result, client = self.run_code(
                    copy.deepcopy(candidate), "A", responses=[response])
                self.assertEqual(result["questions"][0]["status"], "needs_review")
                self.assertEqual(result["questions"][0]["review_error"],
                                 expected_error)
                self.assertEqual(
                    [receipt["stage"] for receipt in client.usage],
                    ["review_code_distinctiveness"],
                )
                self.assertFalse(result.get("revisions"))

    def test_none_value_can_use_real_change_or_failure_basis_without_keyword_ban(self):
        cases = [
            (
                "A",
                self.candidate(
                    "seed 的历史变化是什么？",
                    "seed 从 0 改为 None。", ["e1", "e2"],
                    qa_type="history_tracking", track="history_core"),
            ),
            (
                "B",
                self.candidate(
                    "失败反馈对 seed=None 时的处理约束是什么？",
                    "失败反馈要求 seed=None 时仍最多重试两次。", ["e3", "e4"],
                    qa_type="failure_diagnosis"),
            ),
        ]
        for basis, candidate in cases:
            with self.subTest(basis=basis):
                result, client = self.run_code(candidate, basis)
                self.assertEqual(result["questions"][0]["status"], "approved")
                self.assertEqual(
                    [receipt["stage"] for receipt in client.usage],
                    ["review_code_distinctiveness", "review_relevance",
                     "review_atomicity", "review_completeness",
                     "review_evidence"],
                )

    def test_general_simple_review_call_count_includes_relevance(self):
        candidate = {
            "id": "general-q1",
            "candidate_id": "general-q1",
            "model_id": "q1",
            "qa_mode": "general",
            "type": "single-hop",
            "question": "端口从旧值改为新值后是多少？",
            "answer_points": [{"text": "端口从 18080 改为 18081。",
                               "sources": ["e1", "e2"]}],
            "forbidden_points": [{"text": "端口仍为 18080。", "sources": ["e1"]}],
        }
        client = ScriptedClient([
            atomicity_review(), completeness_review(), evidence_review(),
        ])
        result = review_candidates(
            self.scope, self.facts, [candidate], client,
            qa_mode="general", review_mode="simple", allow_repair=False)
        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual(
            [receipt["stage"] for receipt in client.usage],
            ["review_relevance", "review_atomicity", "review_completeness",
             "review_evidence"],
        )
        self.assertNotIn("review_code_distinctiveness", [
            receipt["stage"] for receipt in client.usage])


if __name__ == "__main__":
    unittest.main()

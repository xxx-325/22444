"""Offline regressions for simple-review prompt isolation and lifecycle."""

import copy
import unittest

from dialogue_benchmark.llm import parse_text_response, review_candidates
from tests.simple_test_helpers import maybe_relevance_response


class ScriptedClient:
    """Return scripted model documents while retaining request projections."""

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
        if callable(response):
            response = response(prompt, payload)
        if isinstance(response, str):
            return parse_text_response(response)
        return copy.deepcopy(response)


def atomicity_review(point_evidence):
    return """REVIEW q1
review_contract: simple_atomicity_v1
point_atomicity: %s
END_REVIEW""" % point_evidence


def completeness_review(value="complete", missing="none"):
    return """REVIEW q1
review_contract: simple_v1
completeness: %s
missing: %s
END_REVIEW""" % (value, missing)


def evidence_review(point_evidence):
    return """REVIEW q1
review_contract: simple_v1
point_evidence: %s
END_REVIEW""" % point_evidence


class SimplePromptContractTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "cutoff": 4,
            "dialogue": [
                {"id": "m1", "kind": "message", "role": "user", "order": 1,
                 "stage_id": "s1", "text": "端口由 18080 改为 18081。"},
                {"id": "m2", "kind": "message", "role": "assistant", "order": 2,
                 "stage_id": "s1", "text": "端口不可用时最多重试两次并返回失败。"},
                {"id": "m3", "kind": "message", "role": "assistant", "order": 3,
                 "stage_id": "s1", "text": "审核需要保留反证。"},
            ],
            "events": [],
            "versions": [],
            "stages": [{"id": "s1"}],
            "review_guard_sources": ["m3"],
            "review_guard_complete": True,
            "model_request_chars": 60000,
            "max_context_chars": 60000,
        }
        self.facts = [
            {"id": "f1", "statement": "端口由 18080 改为 18081。", "sources": ["m1"]},
            {"id": "f2", "statement": "端口不可用时最多重试两次并返回失败。", "sources": ["m2"]},
        ]
        self.candidate = {
            "id": "general_g2_q1",
            "candidate_id": "general_g2_q1",
            "model_id": "q1",
            "qa_mode": "general",
            "type": "single-hop",
            "question": "端口如何变化，端口不可用时如何处理？",
            "answer_points": [
                {"text": "端口由 18080 改为 18081。", "sources": ["m1"]},
                {"text": "端口不可用时最多重试两次并返回失败。", "sources": ["m2"]},
            ],
            "forbidden_points": [
                {"text": "端口保持 18080。", "sources": ["m1"]},
            ],
        }
        self.generation_context = {
            "prompt": "ORIGINAL QA INSTRUCTION",
            "payload": {
                "facts": [{"text": "端口由 18080 改为 18081。", "materials": ["资料1"]}],
                "materials": [
                    {"reference": "资料1", "text": "端口由 18080 改为 18081。"},
                    {"reference": "资料2", "text": "端口不可用时最多重试两次并返回失败。"},
                ],
                "relations": [],
            },
            "ref_to_source": {"资料1": "m1", "资料2": "m2"},
        }

    def _run(self, responses, candidate=None, allow_repair=False, checkpoint=None):
        client = ScriptedClient(responses)
        result = review_candidates(
            self.scope, self.facts, [candidate or self.candidate], client,
            qa_mode="general", review_mode="simple",
            allow_repair=allow_repair, checkpoint=checkpoint,
            generation_context=self.generation_context)
        return result, client

    def test_atomicity_prompt_is_local_and_supplement_reviews_only_missing_a2(self):
        checkpoints = {}
        supplement_payloads = []

        def supplement(_prompt, payload):
            supplement_payloads.append(copy.deepcopy(payload))
            return {
                "reviews": [{
                    "id": "q1",
                    "review_contract": "simple_v1",
                    "point_evidence": "A2=supported@资料2",
                }],
            }

        result, client = self._run([
            atomicity_review("A1=single;A2=single;F1=single"),
            completeness_review(),
            evidence_review("A1=supported@资料1;F1=contradicted@资料1"),
            supplement,
        ], checkpoint=lambda name, data: checkpoints.setdefault(
            name, copy.deepcopy(data)))

        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual(
            [receipt["stage"] for receipt in client.usage],
            ["review_relevance", "review_atomicity", "review_completeness",
             "review_evidence", "review_evidence_supplement"],
        )
        self.assertEqual(len(supplement_payloads), 1)
        supplement_candidate = supplement_payloads[0]["candidates"][0]
        self.assertEqual(supplement_payloads[0]["missing_review_ids"], ["A2"])
        self.assertEqual(
            [point["review_id"] for point in supplement_candidate["answer_points"]],
            ["A2"],
        )
        self.assertEqual(supplement_candidate["forbidden_points"], [])
        self.assertIn("atomicity-review.json", checkpoints)
        self.assertIn("evidence-review-supplement.json", checkpoints)
        self.assertIn("evidence-review-merged.json", checkpoints)
        self.assertEqual(
            result["questions"][0]["atomicity_review"]["review_contract"],
            "simple_atomicity_v1",
        )

        atomic_prompt, atomic_payload = client.calls[1]
        self.assertIn("simple_atomicity_v1", atomic_prompt)
        self.assertIn("point_atomicity", atomic_prompt)
        self.assertEqual(
            set(atomic_payload["candidates"][0]),
            {"id", "immutable_question", "answer_points", "forbidden_points"},
        )
        self.assertEqual(
            [point["review_id"] for point in atomic_payload["candidates"][0][
                "answer_points"]],
            ["A1", "A2"],
        )
        self.assertEqual(
            [point["review_id"] for point in atomic_payload["candidates"][0][
                "forbidden_points"]],
            ["F1"],
        )
        self.assertNotIn("facts", atomic_payload)
        self.assertNotIn("materials", atomic_payload)

    def test_evidence_supplement_is_one_shot_after_atomicity(self):
        result, client = self._run([
            atomicity_review("A1=single;A2=single;F1=single"),
            completeness_review(),
            evidence_review("A1=supported@资料1;F1=contradicted@资料1"),
            # The missing A2 remains unresolved; no second supplement is allowed.
            evidence_review("A1=supported@资料1"),
        ])

        self.assertEqual(result["questions"][0]["status"], "needs_review")
        self.assertEqual(len(client.calls), 5)
        self.assertEqual(
            [receipt["stage"] for receipt in client.usage],
            ["review_relevance", "review_atomicity", "review_completeness",
             "review_evidence", "review_evidence_supplement"],
        )

    def test_missing_completeness_repairs_by_append_and_reruns_atomicity(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["question"] = "端口如何变化以及端口不可用时如何处理？"
        candidate["answer_points"] = candidate["answer_points"][:1]
        candidate["forbidden_points"] = []
        repair_response = """QA q1
QUESTION: 端口如何变化以及端口不可用时如何处理？
ANSWER_POINT: 端口由 18080 改为 18081。 || SOURCES: 资料1
ANSWER_POINT: 端口不可用时最多重试两次并返回失败。 || SOURCES: 资料2
END_QA"""

        result, client = self._run([
            atomicity_review("A1=single"),
            completeness_review("missing", "端口不可用时的处理"),
            evidence_review("A1=supported@资料1"),
            repair_response,
            atomicity_review("A1=single;A2=single"),
            completeness_review(),
            evidence_review("A1=supported@资料1;A2=supported@资料2"),
        ], candidate=candidate, allow_repair=True)

        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertTrue(result["questions"][0]["repair_attempted"])
        self.assertEqual(
            [point["text"] for point in result["questions"][0]["answer_points"]],
            ["端口由 18080 改为 18081。", "端口不可用时最多重试两次并返回失败。"],
        )
        self.assertEqual(
            [receipt["stage"] for receipt in client.usage],
            ["review_relevance", "review_atomicity", "review_completeness",
             "review_evidence", "repair", "review_relevance",
             "review_atomicity", "review_completeness", "review_evidence"],
        )


if __name__ == "__main__":
    unittest.main()

"""Offline regressions for one-shot simple evidence-review replenishment."""

import copy
import json
import unittest

from dialogue_benchmark.llm import parse_text_response, review_candidates
from tests.simple_test_helpers import maybe_relevance_response


class FakeReviewClient:
    """Return scripted model responses while retaining every request payload."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.usage = []

    def ask(self, prompt, payload):
        # Target routing is exercised separately in test_memory_types.
        if "review_contract: target_v1" in prompt:
            return {"reviews": [{"id": "q1", "review_contract": "target_v1",
                                 "target_alignment": "aligned"}]}
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


class MissingReviewTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "cutoff": 3,
            "dialogue": [
                {"id": "m1", "kind": "message", "role": "user", "order": 1,
                 "stage_id": "s1", "text": "配置必须支持 yaml。"},
                {"id": "m2", "kind": "message", "role": "user", "order": 2,
                 "stage_id": "s1", "text": "旧格式仍需兼容。"},
                {"id": "m3", "kind": "message", "role": "assistant", "order": 3,
                 "stage_id": "s1", "text": "审核时还必须检查兼容性证据。"},
            ],
            "events": [],
            "versions": [],
            "stages": [{"id": "s1"}],
            "review_guard_sources": ["m3"],
            "review_guard_complete": True,
            "review_guard_reason": "complete",
            "model_request_chars": 60000,
            "max_context_chars": 60000,
        }
        self.facts = [
            {"id": "f1", "statement": "配置必须支持 yaml", "sources": ["m1"]},
            {"id": "f2", "statement": "旧格式仍需兼容", "sources": ["m2"]},
        ]
        self.candidate = {
            "id": "general_g7_q1",
            "candidate_id": "general_g7_q1",
            "model_id": "model-q1",
            "qa_mode": "general",
            "type": "constraint_followthrough",
            "question": "配置必须支持什么格式，并保留什么兼容性？",
            "fact_ids": ["f1", "f2"],
            "answer_points": [
                {"text": "配置必须支持 yaml。", "sources": ["m1"]},
                {"text": "旧格式仍需兼容。", "sources": ["m2"]},
            ],
            "forbidden_points": [
                {"text": "配置只支持 JSON。", "sources": ["m1"]},
            ],
        }
        self.generation_context = {
            "prompt": "ORIGINAL QA INSTRUCTION",
            "payload": {
                "facts": [{"text": "配置必须支持 yaml", "materials": ["资料1"]}],
                "materials": [
                    {"reference": "资料1", "text": "配置必须支持 yaml。"},
                    {"reference": "资料2", "text": "旧格式仍需兼容。"},
                ],
                "relations": [],
            },
            "ref_to_source": {"资料1": "m1", "资料2": "m2"},
        }

    @staticmethod
    def completeness(value="complete", missing="none"):
        return """REVIEW q1
review_contract: simple_v1
completeness: %s
missing: %s
END_REVIEW""" % (value, missing)

    @staticmethod
    def evidence(value):
        return """REVIEW q1
review_contract: simple_v1
point_evidence: %s
END_REVIEW""" % value

    @staticmethod
    def atomicity(candidate):
        point_ids = (["A%d" % (index + 1)
                      for index in range(len(candidate["answer_points"]))]
                     + ["F%d" % (index + 1)
                        for index in range(len(candidate["forbidden_points"]))])
        return """REVIEW q1
review_contract: simple_atomicity_v1
point_atomicity: %s
END_REVIEW""" % ";".join(
            point_id + "=single" for point_id in point_ids)

    @staticmethod
    def _candidate_payload(call):
        return call[1]["candidates"][0]

    @staticmethod
    def _supplement_for_missing_point(_prompt, payload):
        """Answer the one point actually present in a supplement payload."""
        candidate = payload["candidates"][0]
        points = candidate.get("answer_points", []) + candidate.get(
            "forbidden_points", [])
        if len(points) != 1:
            return {"reviews": []}
        point = points[0]
        review_id = point["review_id"]
        reference = point.get("sources", ["资料1"])[0]
        status = "contradicted" if review_id.startswith("F") else "supported"
        return {"reviews": [{
            "id": "q1",
            "review_contract": "simple_v1",
            "point_evidence": "%s=%s@%s" % (review_id, status, reference),
        }]}

    def _run(self, responses, candidate=None, allow_repair=False,
             review_mode="simple", checkpoint=None, scope=None):
        selected = candidate or self.candidate
        if review_mode == "simple":
            responses = [self.atomicity(selected)] + list(responses)
        client = FakeReviewClient(responses)
        result = review_candidates(
            scope or self.scope, self.facts, [selected], client,
            qa_mode="general", review_mode=review_mode,
            allow_repair=allow_repair, checkpoint=checkpoint,
            generation_context=self.generation_context)
        return result, client

    def test_complete_evidence_does_not_trigger_supplement(self):
        result, client = self._run([
            self.completeness(),
            self.evidence("A1=supported@资料1;A2=supported@资料2;"
                          "F1=contradicted@资料1"),
        ])

        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual(len(client.calls), 4)
        self.assertEqual([item["stage"] for item in client.usage], [
            "review_relevance", "review_atomicity", "review_completeness",
            "review_evidence"])

    def test_missing_a2_supplement_is_narrow_and_can_approve(self):
        checkpoints = {}

        def checkpoint(name, data):
            checkpoints[name] = copy.deepcopy(data)

        result, client = self._run([
            self.completeness(),
            self.evidence("A1=supported@资料1;F1=contradicted@资料1"),
            self._supplement_for_missing_point,
        ], checkpoint=checkpoint)

        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual(len(client.calls), 5)
        supplement = self._candidate_payload(client.calls[4])
        self.assertEqual([point["review_id"] for point in
                          supplement["answer_points"]], ["A2"])
        self.assertEqual(supplement["answer_points"][0]["immutable_text"],
                         self.candidate["answer_points"][1]["text"])
        self.assertEqual(supplement.get("forbidden_points", []), [])
        self.assertEqual(supplement["id"], "q1")

        materials = json.dumps(client.calls[4][1].get("materials", []),
                               ensure_ascii=False)
        self.assertIn("旧格式仍需兼容。", materials)
        self.assertIn("审核时还必须检查兼容性证据。", materials)
        self.assertNotIn("配置必须支持 yaml。", materials)
        evidence = result["questions"][0]["evidence_review"]
        self.assertIn("A2=supported@m2", evidence["point_evidence"])
        self.assertIn("completeness-review.json", checkpoints)
        self.assertIn("evidence-review.json", checkpoints)
        self.assertEqual(checkpoints["evidence-review-missing.json"]["missing_review_ids"],
                         ["A2"])
        self.assertIn("evidence-review-supplement.json", checkpoints)
        self.assertIn("evidence-review-merged.json", checkpoints)
        self.assertEqual(
            set(checkpoints["evidence-review-merged.json"]["reviews"][0][
                "point_evidence"].split(";")),
            {"A1=supported@m1", "A2=supported@m2", "F1=contradicted@m1"})

    def test_existing_false_judgment_survives_missing_point_supplement(self):
        result, _client = self._run([
            self.completeness(),
            self.evidence("A1=contradicted@资料1;F1=contradicted@资料1"),
            self._supplement_for_missing_point,
        ])

        self.assertNotIn("approved", [item.get("status")
                                       for item in result.get("questions", [])])
        serialized = json.dumps(result, ensure_ascii=False)
        self.assertIn("A1=contradicted@m1", serialized)

    def test_supplement_still_missing_is_needs_review_and_one_shot(self):
        result, client = self._run([
            self.completeness(),
            self.evidence("A1=supported@资料1;F1=contradicted@资料1"),
            self.evidence("A1=supported@资料1"),
        ])

        self.assertEqual(result["questions"][0]["status"], "needs_review")
        self.assertEqual(len(client.calls), 5)

    def test_supplement_exception_is_needs_review_and_one_shot(self):
        result, client = self._run([
            self.completeness(),
            self.evidence("A1=supported@资料1;F1=contradicted@资料1"),
            RuntimeError("synthetic supplement failure"),
        ])

        self.assertEqual(result["questions"][0]["status"], "needs_review")
        self.assertEqual(len(client.calls), 5)
        self.assertEqual(result["stage_status"]["review"], "failed")
        self.assertEqual(result["stage_errors"][0]["stage"],
                         "review_evidence_supplement")

    def test_invalid_initial_review_does_not_trigger_supplement(self):
        invalid_reviews = {
            "duplicate point": "A1=supported@资料1;A1=supported@资料1;"
                               "F1=contradicted@资料1",
            "extra point": "A1=supported@资料1;F1=contradicted@资料1;"
                            "A3=supported@资料2",
            "invalid source": "A1=supported@资料999;F1=contradicted@资料1",
        }
        for label, point_evidence in invalid_reviews.items():
            with self.subTest(label=label):
                result, client = self._run([
                    self.completeness(), self.evidence(point_evidence),
                    self._supplement_for_missing_point,
                ])
                self.assertEqual(result["questions"][0]["status"],
                                 "needs_review")
                self.assertEqual(len(client.calls), 4)

    def test_supplement_cannot_override_an_existing_point(self):
        def malicious_supplement(_prompt, payload):
            answer_ref = payload["candidates"][0]["answer_points"][0]["sources"][0]
            return {"reviews": [{
                "id": "q1",
                "review_contract": "simple_v1",
                "point_evidence": (
                    "A1=supported@资料1;A2=supported@%s;F1=supported@资料1"
                    % answer_ref),
            }]}

        result, _client = self._run([
            self.completeness(),
            self.evidence("A1=contradicted@资料1;F1=contradicted@资料1"),
            malicious_supplement,
        ])

        self.assertNotIn("approved", [item.get("status")
                                       for item in result.get("questions", [])])
        self.assertIn("A1=contradicted@m1",
                      json.dumps(result, ensure_ascii=False))

    def test_source_references_and_local_q1_are_stable_across_supplement(self):
        result, client = self._run([
            self.completeness(),
            self.evidence("A1=supported@资料1;F1=contradicted@资料1"),
            self._supplement_for_missing_point,
        ])

        for _prompt, payload in client.calls[1:]:
            self.assertEqual(payload["candidates"][0]["id"], "q1")
        supplement = self._candidate_payload(client.calls[4])
        self.assertEqual(supplement["answer_points"][0]["sources"],
                         ["资料1"])
        self.assertEqual(result["questions"][0]["id"], self.candidate["id"])
        self.assertEqual(result["questions"][0]["evidence_review"]["returned_id"],
                         "q1")

    def test_each_candidate_has_its_own_one_shot_supplement(self):
        second = copy.deepcopy(self.candidate)
        second.update(id="general_g7_q2", candidate_id="general_g7_q2",
                      model_id="model-q2")
        client = FakeReviewClient([
            response
            for _ in range(2)
            for response in (
                self.atomicity(self.candidate),
                self.completeness(),
                self.evidence("A1=supported@资料1;F1=contradicted@资料1"),
                self._supplement_for_missing_point,
            )
        ])
        result = review_candidates(
            self.scope, self.facts, [self.candidate, second], client,
            qa_mode="general", review_mode="simple", allow_repair=False)

        self.assertEqual([item["status"] for item in result["questions"]],
                         ["approved", "approved"])
        self.assertEqual(len(client.calls), 10)
        supplement_payloads = [
            self._candidate_payload(client.calls[index])
            for index in (4, 9)
        ]
        self.assertTrue(all(
            [point["review_id"] for point in payload["answer_points"]] == ["A2"]
            for payload in supplement_payloads))

    def test_repair_re_review_reuses_candidate_supplement_budget(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["answer_points"] = candidate["answer_points"][:1]

        def repair_response(_prompt, _payload):
            return """QA q1
QUESTION: 配置必须支持什么格式，并保留什么兼容性？
ANSWER_POINT: 配置必须支持 yaml。 || SOURCES: 资料1
ANSWER_POINT: 旧格式仍需兼容。 || SOURCES: 资料2
FORBIDDEN_POINT: 配置只支持 JSON。 || SOURCES: 资料1
END_QA"""

        responses = [
            self.completeness("missing", "兼容性"),
            self.evidence("A1=supported@资料1"),
            self._supplement_for_missing_point,
            repair_response,
            self.atomicity(dict(candidate, answer_points=[
                candidate["answer_points"][0],
                {"text": "旧格式仍需兼容。", "sources": ["m2"]},
            ])),
            self.completeness(),
            self.evidence("A1=supported@资料1;F1=contradicted@资料1"),
            self._supplement_for_missing_point,
        ]
        result, client = self._run(responses, candidate=candidate,
                                   allow_repair=True)

        self.assertEqual(result["questions"][0]["status"], "needs_review")
        # The first supplement repairs the missing F1 review.  The recursive
        # review after QA repair may not spend a second supplement on A2.
        self.assertEqual(len(client.calls), 10)
        supplement_payloads = [
            payload["candidates"][0]
            for _prompt, payload in client.calls
            if len(payload.get("candidates", [])) == 1
            and [point["review_id"] for point in
                 payload["candidates"][0].get("answer_points", [])] == []
            and [point["review_id"] for point in
                 payload["candidates"][0].get("forbidden_points", [])]
        ]
        self.assertEqual(len(supplement_payloads), 1)

    def test_verification_reuse_supplement_keeps_full_range_and_guards_other_materials(self):
        scope = copy.deepcopy(self.scope)
        scope["full_range_covered"] = True
        scope["full_range_required"] = True
        scope["dialogue"].append({
            "id": "m4", "kind": "message", "role": "user", "order": 4,
            "stage_id": "s1", "text": "范围内还记录了另一项配置背景。",
        })
        candidate = copy.deepcopy(self.candidate)
        candidate["type"] = "verification_reuse"

        result, client = self._run([
            self.completeness(),
            self.evidence("A1=supported@资料1;F1=contradicted@资料1"),
            self._supplement_for_missing_point,
        ], candidate=candidate, scope=scope)

        self.assertEqual(result["questions"][0]["status"], "approved")
        supplement_payload = self._candidate_payload(client.calls[4])
        self.assertEqual([point["review_id"] for point in
                          supplement_payload["answer_points"]], ["A2"])
        self.assertEqual(supplement_payload.get("forbidden_points", []), [])
        payload = client.calls[4][1]
        materials = payload["materials"]
        material_blob = json.dumps(materials, ensure_ascii=False)
        for text in (
                "配置必须支持 yaml。", "旧格式仍需兼容。",
                "审核时还必须检查兼容性证据。", "范围内还记录了另一项配置背景。"):
            self.assertIn(text, material_blob)

        material_refs = {item["reference"] for item in materials}
        a2_refs = set(supplement_payload["answer_points"][0]["sources"])
        guard_refs = set(payload["counterevidence_guard"]["materials"])
        self.assertEqual(a2_refs, {"资料2"})
        self.assertTrue(material_refs - a2_refs <= guard_refs)
        self.assertTrue(payload["counterevidence_guard"]["complete"])

    def test_legacy_single_review_still_approves_without_supplement(self):
        candidate = copy.deepcopy(self.candidate)
        candidate.update(
            difficulty="easy", difficulty_reason="直接给出格式约束",
            memory_requirement="恢复配置兼容要求",
            use_case="开发者实现配置解析时确认格式与兼容范围",
            answer_target="配置格式与兼容性",
            # Legacy structured review still requires the semantic rationale
            # fields that are not part of the simple contract.
        )
        review = {
            "id": "q1",
            "review_contract": "structured_v2",
            "evidence_supported": True,
            "version_consistent": True,
            "unambiguous": True,
            "difficulty_justified": True,
            "not_answer_leaking": True,
            "natural_wording": True,
            "practical_useful": True,
            "answer_complete": True,
            "atomic_points_correct": True,
            "type_correct": True,
            "recommended_type": "constraint_followthrough",
            "type_basis": "一条讨论阶段中的直接约束",
            "useful_task": "实现配置解析",
            "useful_decision": "确定支持格式和兼容性",
            "answer_effect": "决定解析器兼容范围",
            "necessary_source_ids": "m1,m2",
            "answer_requirements": "R1=配置格式;R2=兼容性",
            "requirement_coverage": "R1=A1;R2=A2",
            "point_claims": "A1.1=配置必须支持 yaml。;A2.1=旧格式仍需兼容。;"
                             "F1.1=配置只支持 JSON。",
            "point_evidence": "A1=supported@m1;A2=supported@m2;F1=contradicted@m1",
            "reason": "结构、用途与证据均通过。",
        }
        result, client = self._run(
            [{"reviews": [review]}], candidate=candidate,
            review_mode="single")

        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.usage[0]["stage"], "review_single")


if __name__ == "__main__":
    unittest.main()

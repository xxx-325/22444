"""The six purposes route generation, review, and derived requirements together."""

import copy
import tempfile
import unittest
from pathlib import Path

from dialogue_benchmark.cli import _build_parser, _parse_options
from dialogue_benchmark.fact_index import build_evidence_groups, build_evidence_index, static_evidence_check
from dialogue_benchmark.llm import generate_from_facts, review_candidates, parse_text_response
from dialogue_benchmark.protocol import QA_TYPES, QA_TYPE_GUIDANCE, TASK_TYPE_GUIDANCE
from dialogue_benchmark.task_eval.artifacts import qa_inputs, save
from dialogue_benchmark.task_eval.prompts import task_direction


class Client:
    def __init__(self, alignment="aligned", repair=False):
        self.alignment, self.repair = alignment, repair
        self.calls, self.usage = [], []

    def ask(self, prompt, payload):
        self.calls.append((prompt, copy.deepcopy(payload)))
        self.usage.append({"status": "completed"})
        if "Correct exactly the supplied review_issue once" in prompt:
            self.alignment = "aligned"
            return parse_text_response("QA q1\nQUESTION: 扩展导出时要继续遵守哪项限制？\n"
                                       "ANSWER_POINT: 导出必须保留空值。 || SOURCES: 资料1\nEND_QA")
        if "Return exactly two lines when a grounded focus exists" in prompt:
            return {"focus": {"text": "确认扩展导出时仍适用的空值限制", "sources": ["资料1"]}}
        if "review_contract: target_v1" in prompt:
            if isinstance(self.alignment, BaseException):
                raise self.alignment
            return {"reviews": [{"id": "q1", "review_contract": "target_v1",
                                 "target_alignment": self.alignment}]}
        if "simple_relevance_v1" in prompt:
            value = {"review_contract": "simple_relevance_v1", "point_relevance": "A1=direct"}
        elif "simple_atomicity_v1" in prompt:
            value = {"review_contract": "simple_atomicity_v1", "point_atomicity": "A1=single"}
        elif "fully answer every requested subject" in prompt:
            value = {"review_contract": "simple_v1", "completeness": "complete", "missing": "none"}
        elif "point_evidence:" in prompt:
            value = {"review_contract": "simple_v1", "point_evidence": "A1=supported@资料1"}
        else:
            return parse_text_response("QA q1\nQUESTION: 扩展导出时要继续遵守哪项限制？\n"
                                       "ANSWER_POINT: 导出必须保留空值。 || SOURCES: 资料1\nEND_QA")
        return {"reviews": [dict(value, id="q1")]}


class MemoryTypeTests(unittest.TestCase):
    def setUp(self):
        self.scope = {"cutoff": 1, "dialogue": [
            {"id": "message", "kind": "message", "role": "user", "order": 1,
             "text": "导出必须保留空值。"}], "events": [], "versions": [],
            "model_request_chars": 60000, "review_guard_complete": True}
        self.facts = [{"id": "fact", "statement": "导出必须保留空值", "sources": ["message"]}]
        self.question = {"id": "stable-q", "type": "constraint_followthrough",
                         "question": "扩展导出时要继续遵守哪项限制？",
                         "answer_points": [{"text": "导出必须保留空值。", "sources": ["message"]}],
                         "forbidden_points": [], "fact_ids": ["fact"]}

    def test_six_types_have_generation_and_task_routes(self):
        self.assertEqual(len(QA_TYPES), 6)
        self.assertEqual(QA_TYPES, set(TASK_TYPE_GUIDANCE))
        for kind in QA_TYPES:
            client = Client()
            result = generate_from_facts(self.scope, self.facts, client, qa_mode="general", target_type=kind)
            self.assertEqual(result["questions"][0]["type"], kind)
            self.assertIn(QA_TYPE_GUIDANCE[kind], client.calls[0][0])
            self.assertIn(QA_TYPE_GUIDANCE[kind], client.calls[1][0])
            self.assertNotIn(kind, client.calls[0][0])
            self.assertIn(TASK_TYPE_GUIDANCE[kind], task_direction(kind))

    def test_both_tracks_review_the_fixed_purpose_without_relabeling(self):
        for mode in ("general", "code"):
            for kind in QA_TYPES:
                with self.subTest(mode=mode, kind=kind):
                    client = Client()
                    q = dict(self.question, qa_mode=mode, type=kind, difficulty="hard")
                    result = review_candidates(self.scope, self.facts, [q], client,
                                               qa_mode=mode, review_mode="simple", allow_repair=False)
                    self.assertEqual(result["questions"][0]["status"], "approved")
                    self.assertEqual(result["questions"][0]["type"], kind)
                    self.assertEqual(result["questions"][0]["difficulty"], "hard")
                    self.assertEqual([r["stage"] for r in client.usage], [
                        "review_target", "review_relevance", "review_atomicity",
                        "review_completeness", "review_evidence"])
                    self.assertIn(QA_TYPE_GUIDANCE[kind], client.calls[0][0])
                    self.assertNotIn("answer_basis", client.calls[0][0])

    def test_target_mismatch_rejects_and_uncertainty_never_approves(self):
        for alignment in ("drifted", "mixed", "uncertain", "invalid", OSError("offline")):
            with self.subTest(alignment=str(alignment)):
                client = Client(alignment)
                result = review_candidates(self.scope, self.facts, [dict(self.question, qa_mode="code")],
                                           client, qa_mode="code", review_mode="simple", allow_repair=False)
                if alignment in ("drifted", "mixed"):
                    self.assertFalse(result["questions"])
                    self.assertEqual(result["rejected"][0]["reason"], "answer_target_mismatch")
                else:
                    self.assertEqual(result["questions"][0]["status"], "needs_review")
                self.assertEqual(len(client.calls), 1)

    def test_target_repair_preserves_identity_and_runs_all_checks(self):
        client = Client("drifted")
        context = {"prompt": "ORIGINAL", "payload": {"materials": [
            {"reference": "资料1", "text": "导出必须保留空值。"}]},
            "ref_to_source": {"资料1": "message"}}
        result = review_candidates(self.scope, self.facts, [dict(self.question, qa_mode="general")],
                                   client, qa_mode="general", review_mode="simple", generation_context=context)
        self.assertEqual(result["questions"][0]["status"], "approved")
        self.assertEqual(result["questions"][0]["id"], "stable-q")
        self.assertEqual(len(result["revisions"]), 1)
        self.assertEqual([r["stage"] for r in client.usage], ["review_target", "repair", "review_target",
                         "review_relevance", "review_atomicity", "review_completeness", "review_evidence"])

    def test_recorded_conditions_nominate_all_six_types_in_both_tracks(self):
        statements = {
            "constraint_followthrough": "用户要求 export 必须保留空值",
            "correction_update": "用户纠正 export 规则：之前丢弃空值，后来改为保留空值",
            "external_state_application": "用户侧 production 运行 export 时只支持 EU 编码",
            "failure_avoidance": "之前 export 使用 ASCII 实际运行失败，输入为 EU 文本，后来改用 EU 编码",
            "verification_reuse": "export 的 EU 输入测试通过，验证结果确认能保留空值",
            "compatibility_preservation": "新 export 必须兼容旧客户端的空值行为",
        }
        for mode in ("general", "code"):
            for kind, statement in statements.items():
                with self.subTest(mode=mode, kind=kind):
                    facts = [dict(self.facts[0], statement=statement)]
                    index = build_evidence_index(facts, [self.scope], mode)
                    groups = build_evidence_groups(facts, [self.scope], mode, {kind}, 1,
                                                   evidence_index=index, static_selection=True)
                    self.assertEqual(len(groups), 1)
                    self.assertEqual(static_evidence_check(groups[0], index, kind)["status"], "supported")

    def test_old_types_are_not_cli_or_task_inputs(self):
        parser = _build_parser()
        for old in ("single-hop", "multi-hop", "temporal", "open-domain", "adversarial",
                    "fact_recall", "history_tracking", "behavior_inference", "failure_diagnosis"):
            with self.subTest(old=old):
                with self.assertRaises(SystemExit):
                    _parse_options(parser.parse_args(["in.json", "--output", "out", "--qa-mode", "general",
                                                      "--general-types", old]), parser)
                with tempfile.TemporaryDirectory() as directory:
                    save(Path(directory) / "qa-public.json", {"questions": [{"type": old}]})
                    with self.assertRaises(ValueError):
                        qa_inputs(directory)


if __name__ == "__main__":
    unittest.main()

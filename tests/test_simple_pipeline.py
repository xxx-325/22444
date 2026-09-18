"""Offline contracts for simple target binding and static difficulty metadata."""

import copy
import unittest

from dialogue_benchmark.fact_index import _relation_metadata
from dialogue_benchmark.llm import generate_from_facts
from tests.simple_test_helpers import maybe_focus_response


class CaptureClient:
    """Return a fixed parsed response without constructing a provider client."""

    def __init__(self, response):
        self.response = copy.deepcopy(response)
        self.prompts = []
        self.payloads = []
        self.usage = []

    def ask(self, prompt, payload):
        self.prompts.append(prompt)
        self.payloads.append(copy.deepcopy(payload))
        self.usage.append({"status": "completed"})
        focus = maybe_focus_response(prompt, payload)
        if focus is not None:
            return focus
        return copy.deepcopy(self.response)


class SimplePipelineTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "cutoff": 1,
            "dialogue": [{
                "id": "e1", "kind": "message", "role": "user", "order": 1,
                "stage_id": "s1", "text": "配置必须支持 yaml。",
            }],
            "events": [], "versions": [], "stages": [{"id": "s1"}],
            "evidence_group": {"target_types": ["single-hop"]},
            "model_request_chars": 60000,
            "max_context_chars": 60000,
        }
        self.facts = [{
            "id": "f1", "qa_mode": "general", "statement": "配置必须支持 yaml",
            "sources": ["e1"],
        }]
        self.response = {
            "questions": [{
                "id": "q1", "fact_ids": ["f1"],
                "question": "配置必须支持什么格式？",
                "answer_points": [{"text": "配置必须支持 yaml。", "sources": ["资料1"]}],
                "forbidden_points": [],
            }],
        }

    def test_simple_prompt_binds_one_target_without_annotation_fields(self):
        client = CaptureClient(self.response)
        result = generate_from_facts(
            self.scope, self.facts, client, qa_mode="general",
            allowed_types={"single-hop"}, generation_mode="simple",
            target_type="single-hop")

        self.assertEqual(result["stage_status"]["qa"], "completed")
        candidate = result["questions"][0]
        self.assertEqual(candidate["type"], "single-hop")
        self.assertNotIn("difficulty", candidate)
        self.assertNotIn("track", candidate)
        self.assertNotIn("use_case", candidate)
        prompt = client.prompts[1]
        self.assertIn("SOURCES: 资料1", prompt)
        self.assertNotIn("DIFFICULTY:", prompt)
        self.assertNotIn("TRACK:", prompt)

    def test_explicit_target_must_be_allowed_and_multiple_allowed_types_need_choice(self):
        # A target bound by the caller cannot silently escape its allowed set.
        client = CaptureClient(self.response)
        result = generate_from_facts(
            self.scope, self.facts, client, qa_mode="general",
            allowed_types={"single-hop"}, generation_mode="simple",
            target_type="temporal")
        self.assertEqual(result["stage_status"]["qa"], "failed")
        self.assertFalse(client.payloads)

        client = CaptureClient(self.response)
        result = generate_from_facts(
            self.scope, self.facts, client, qa_mode="general",
            allowed_types={"single-hop", "temporal"}, generation_mode="simple")
        self.assertEqual(result["stage_status"]["qa"], "failed")
        self.assertFalse(client.payloads)

    def test_static_relation_distance_mapping_has_no_annotation_call(self):
        def metadata(distance):
            graph = {"n0": []}
            for index in range(distance):
                left, right = "n%d" % index, "n%d" % (index + 1)
                graph.setdefault(left, []).append((right, 1, "explicit"))
                graph.setdefault(right, []).append((left, 1, "explicit"))
            infos = [{"fact": {"id": "f0", "sources": ["n0"]}}]
            if distance:
                infos.append({"fact": {
                    "id": "f%d" % distance, "sources": ["n%d" % distance],
                }})
            source_index = {node: {"order": index}
                            for index, node in enumerate(graph)}
            return _relation_metadata(
                infos, {"direct_graph": graph, "source_index": source_index})

        for distance, expected in (
                (0, "easy"), (1, "easy"), (2, "medium"), (3, "hard")):
            with self.subTest(distance=distance):
                result = metadata(distance)
                self.assertEqual(result["max_distance"], distance)
                self.assertEqual(result["difficulty"], expected)

        unknown = _relation_metadata(
            [{"fact": {"id": "f0", "sources": ["n0"]}},
             {"fact": {"id": "f9", "sources": ["n9"]}}],
            {"direct_graph": {"n0": [], "n9": []},
             "source_index": {"n0": {"order": 0}, "n9": {"order": 9}}})
        self.assertIsNone(unknown["max_distance"])
        self.assertEqual(unknown["difficulty"], "unknown")


if __name__ == "__main__":
    unittest.main()

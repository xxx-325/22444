import unittest

from dialogue_benchmark.general import build_general_scope, build_object_index


class GeneralIndexTests(unittest.TestCase):
    def test_code_objects_index_far_apart_visible_turns(self):
        records = [
            {"id": "e1", "order": 1, "kind": "message", "role": "user",
             "text": "challenge.py 必须保留 Attribute 限制。"},
            {"id": "e2", "order": 2, "kind": "message", "role": "assistant",
             "text": "中间无关讨论。"},
            {"id": "e90", "order": 90, "kind": "message", "role": "user",
             "text": "后来 solve.py 的 join 方案要配合 checker。"},
        ]
        scope = build_general_scope(records, 90, graph={
            "versions": [{"id": "v1", "path": "project/challenge.py",
                          "source": "e1", "code": {"symbols": [{"name": "validate_expression"}]}}]
        })
        files = [item for item in scope["object_index"] if item["object_kind"] == "file"]
        self.assertTrue(any("challenge.py" in item["aliases"] for item in files))
        challenge = next(item for item in files if item["normalized_name"].endswith("challenge.py"))
        self.assertIn("e1", challenge["message_ids"])
        self.assertEqual(scope["dialogue"][0]["stage_id"], "stage-1")
        self.assertNotEqual(scope["dialogue"][0]["stage_id"], scope["dialogue"][-1]["stage_id"])

    def test_index_does_not_link_unmentioned_generic_turn(self):
        index = build_object_index([
            {"id": "e1", "text": "用户要求保留 challenge.py 的限制。", "stage_id": "stage-1"},
            {"id": "e2", "text": "讨论天气和午饭。", "stage_id": "stage-2"},
        ])
        item = next(item for item in index if item["normalized_name"] == "challenge.py")
        self.assertEqual(item["message_ids"], ["e1"])

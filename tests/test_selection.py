import unittest

from dialogue_benchmark.llm import parse_text_response
from dialogue_benchmark.selection import (
    apply_duplicate_decisions,
    deduplicate,
    deduplicate_reviewed,
    near_duplicate_clusters,
    review_duplicate_clusters,
)


def question(qid, target, answer, source="e1", mode="code", status="approved"):
    return {
        "id": qid,
        "qa_mode": mode,
        "status": status,
        "answer_target": target,
        "question": target + "？",
        "answer_points": [{"text": answer, "sources": [source]}],
        "forbidden_points": [],
    }


class SelectionTests(unittest.TestCase):
    def test_early_dedup_keeps_semantic_alternatives(self):
        short = question("short", "确认运行编排的 Agent 时限", "Agent 最长 60 分钟")
        broad = question(
            "broad", "确认运行编排的 Agent 与 Harness 时限",
            "Agent 最长 60 分钟，Harness 最长 30 分钟")
        kept, excluded = deduplicate([short, broad])
        self.assertEqual([item["id"] for item in kept], ["short", "broad"])
        self.assertFalse(excluded)
        self.assertEqual([[item["id"] for item in cluster]
                          for cluster in near_duplicate_clusters(kept)],
                         [["short", "broad"]])

    def test_reviewed_dedup_keeps_approved_candidate(self):
        pending = question(
            "pending", "确认 detail 时间参数在补丁后的表达式",
            "补丁后使用 datetime(2020, 1, number, 1, tzinfo=UTC)",
            status="needs_review")
        approved = question(
            "approved", "确认 detail 时间参数在补丁后的表达式",
            "补丁后使用 datetime(2020, 1, number, 1, tzinfo=UTC)")
        approved["question"] = "补丁以后 detail 使用哪个时间表达式？"
        kept, excluded = deduplicate_reviewed([pending, approved])
        self.assertEqual([item["id"] for item in kept], ["approved"])
        self.assertEqual(excluded[0]["duplicate_of"], "approved")
        self.assertEqual(excluded[0]["reason"], "near_duplicate_answer_target")

    def test_reviewed_dedup_prefers_more_complete_contained_answer(self):
        short = question("short", "确认运行编排时限", "Agent 最长 60 分钟")
        broad = question(
            "broad", "确认运行编排时限", "Agent 最长 60 分钟")
        broad["answer_points"].append(
            {"text": "Harness 最长 30 分钟", "sources": ["e1"]})
        kept, excluded = deduplicate_reviewed([short, broad])
        self.assertEqual([item["id"] for item in kept], ["broad"])
        self.assertEqual(excluded[0]["candidate_id"], "short")

    def test_same_evidence_with_different_targets_is_preserved(self):
        timeout = question("timeout", "决定 Agent 的超时阈值", "Agent 最长 60 分钟")
        owner = question("owner", "决定由哪个组件执行超时", "超时由 Adapter 执行")
        kept, excluded = deduplicate_reviewed([timeout, owner])
        self.assertEqual([item["id"] for item in kept], ["owner", "timeout"])
        self.assertFalse(excluded)

    def test_same_target_at_conflicting_times_is_preserved(self):
        before = question(
            "before", "确认修改前的 detail 时间参数",
            "修改前使用 datetime(2020, 2, 1, tzinfo=UTC)")
        after = question(
            "after", "确认修改后的 detail 时间参数",
            "修改后使用 datetime(2020, 1, number, 1, tzinfo=UTC)")
        kept, excluded = deduplicate_reviewed([before, after])
        self.assertEqual([item["id"] for item in kept], ["after", "before"])
        self.assertFalse(excluded)

    def test_similar_operators_numbers_and_polarity_are_never_auto_deleted(self):
        cases = [
            ("x > 0 时的处理", "x < 0 时的处理", "返回 enabled", "返回 enabled"),
            ("开关状态", "开关状态", "feature enabled", "feature disabled"),
            ("重试次数", "重试次数", "重试 5 次", "重试 50 次"),
        ]
        for index, (left_target, right_target, left_answer, right_answer) in enumerate(cases):
            left = question("left%d" % index, left_target, left_answer)
            right = question("right%d" % index, right_target, right_answer)
            kept, excluded = deduplicate_reviewed([left, right])
            self.assertEqual(len(kept), 2)
            self.assertFalse(excluded)

    def test_same_question_with_conflicting_answers_is_not_an_exact_duplicate(self):
        left = question("left", "确认开关状态", "feature enabled")
        right = question("right", "确认开关状态", "feature disabled")
        kept, excluded = deduplicate([left, right])
        self.assertEqual(len(kept), 2)
        self.assertFalse(excluded)

    def test_small_cluster_review_is_bounded_and_approved_first(self):
        pending = question(
            "pending", "确认 Adapter 的 Agent 硬停止线", "Agent 最长 60 分钟",
            status="needs_review")
        approved = question(
            "approved", "确认 Agent 超时上限", "Agent 最长 60 分钟")

        class Client:
            def __init__(self):
                self.usage = []
                self.calls = []

            def ask(self, prompt, payload):
                # Target routing is exercised separately in test_memory_types.
                if "review_contract: target_v1" in prompt:
                    return {"reviews": [{"id": "q1", "review_contract": "target_v1",
                                         "target_alignment": "aligned"}]}
                self.calls.append(payload)
                self.usage.append({"request_count": 1})
                pair_id = payload["pairs"][0]["id"]
                assert "REVIEW left_id|right_id" in prompt
                assert "END_REVIEW" in prompt
                assert "Do not return JSON" in prompt
                return parse_text_response(
                    "REVIEW %s\n"
                    "same_target: true\n"
                    "same_time: true\n"
                    "same_answer: true\n"
                    "duplicate_of: pending\n"
                    "reason: 同一目标、时点与答案要求\n"
                    "END_REVIEW" % pair_id)

        client = Client()
        reviewed = review_duplicate_clusters([pending, approved], client)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(reviewed["usage"], [{"request_count": 1}])
        self.assertEqual(reviewed["decisions"][0]["candidate_id"], "pending")
        self.assertEqual(reviewed["decisions"][0]["duplicate_of"], "approved")
        kept, excluded = apply_duplicate_decisions(
            [pending, approved], reviewed["decisions"])
        self.assertEqual([item["id"] for item in kept], ["approved"])
        self.assertEqual(excluded[0]["reason"], "reviewed_near_duplicate")

    def test_failed_cluster_review_retains_every_candidate(self):
        left = question("left", "确认 Agent 时限", "Agent 最长 60 分钟")
        right = question("right", "确认 Agent 超时", "Agent 最长 60 分钟")

        class Client:
            usage = []

            def ask(self, prompt, payload):
                # Target routing is exercised separately in test_memory_types.
                if "review_contract: target_v1" in prompt:
                    return {"reviews": [{"id": "q1", "review_contract": "target_v1",
                                         "target_alignment": "aligned"}]}
                raise RuntimeError("offline")

        reviewed = review_duplicate_clusters([left, right], Client())
        self.assertFalse(reviewed["decisions"])
        self.assertEqual(reviewed["errors"][0]["error_type"], "RuntimeError")
        kept, excluded = apply_duplicate_decisions(
            [left, right], reviewed["decisions"])
        self.assertEqual(len(kept), 2)
        self.assertFalse(excluded)

    def test_duplicate_review_reason_conflict_retains_both_candidates(self):
        left = question("left", "确认 Agent 时限", "Agent 最长 60 分钟")
        right = question("right", "确认 Agent 超时", "Agent 最长 60 分钟")

        class Client:
            usage = []

            def ask(self, prompt, payload):
                # Target routing is exercised separately in test_memory_types.
                if "review_contract: target_v1" in prompt:
                    return {"reviews": [{"id": "q1", "review_contract": "target_v1",
                                         "target_alignment": "aligned"}]}
                pair_id = payload["pairs"][0]["id"]
                return parse_text_response(
                    "REVIEW %s\n"
                    "same_target: true\n"
                    "same_time: true\n"
                    "same_answer: true\n"
                    "duplicate_of: left\n"
                    "reason: 两题实际对应不同时间，不能合并\n"
                    "END_REVIEW" % pair_id)

        reviewed = review_duplicate_clusters([left, right], Client())
        self.assertFalse(reviewed["decisions"])
        kept, excluded = apply_duplicate_decisions(
            [left, right], reviewed["decisions"])
        self.assertEqual(len(kept), 2)
        self.assertFalse(excluded)

    def test_duplicate_review_without_reason_retains_both_candidates(self):
        left = question("left", "确认 Agent 时限", "Agent 最长 60 分钟")
        right = question("right", "确认 Agent 超时", "Agent 最长 60 分钟")

        class Client:
            usage = []

            def ask(self, prompt, payload):
                # Target routing is exercised separately in test_memory_types.
                if "review_contract: target_v1" in prompt:
                    return {"reviews": [{"id": "q1", "review_contract": "target_v1",
                                         "target_alignment": "aligned"}]}
                pair_id = payload["pairs"][0]["id"]
                return parse_text_response(
                    "REVIEW %s\n"
                    "same_target: true\n"
                    "same_time: true\n"
                    "same_answer: true\n"
                    "duplicate_of: left\n"
                    "END_REVIEW" % pair_id)

        reviewed = review_duplicate_clusters([left, right], Client())
        self.assertFalse(reviewed["decisions"])


if __name__ == "__main__":
    unittest.main()

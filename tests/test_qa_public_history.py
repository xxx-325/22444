import json
import tempfile
import unittest
from pathlib import Path

from dialogue_benchmark.external import load_external_scopes, external_review_projection, filter_external_facts
from dialogue_benchmark.fact_index import build_evidence_index, static_candidate_labels
from dialogue_benchmark.llm import _evidence_review_request
from dialogue_benchmark.selection import difficulty_balance, type_balance, duplicate_reason


class PublicHistoryTests(unittest.TestCase):
    def scopes(self, records, events):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            path.write_text(json.dumps({"version": 1, "events": events}))
            return load_external_scopes(path, records, len(records), merge_task_events=False)["scopes"]

    def event(self, identity, source, **fields):
        return dict(id=identity, kind="external_observation", memory_kind="M2",
                    source_ids=[source], scope_policy="declared", **fields)

    def record(self, identity, order, text):
        return dict(id=identity, order=order, kind="message", role="user", text=text,
                    source_kind="conversation")

    def test_declared_seed_gets_only_related_global_amendments_and_exception(self):
        records = [self.record("e1", 1, "Customer Acme export includes empty notes."),
                   self.record("e2", 2, "Customer Beta export stays unchanged."),
                   self.record("e3", 3, "Correction: Acme export must omit empty notes only in cycle 7."),
                   self.record("e4", 4, "Acme export exception: signed notes still remain."),
                   self.record("e5", 5, "Correction: unrelated dashboard font is green.")]
        scope, = self.scopes(records, [self.event("seed", "e1")])
        self.assertEqual([row["id"] for row in scope["dialogue"]], ["e1", "e3", "e4"])
        candidate = dict(id="q1", question="What did Acme export require before cycle 7?",
                         answer_points=[dict(text="Include empty notes.", sources=["e1"])])
        projected, audit = external_review_projection(dict(scope=scope, facts=[]), candidate)
        self.assertTrue(audit["complete"])
        prompt, payload, refs = _evidence_review_request(
            projected["scope"], projected["scope"]["review_guard_sources"], [], candidate)
        self.assertIn("Historical questions may use the earlier rule", prompt)
        self.assertIn("stale only inside its stated scope", prompt.replace("\n", " "))
        self.assertIn("e4", refs.values())
        self.assertNotIn("dashboard", str(payload))
        facts = [dict(id="f", statement=records[2]["text"], sources=["e3"])]
        self.assertEqual(len(filter_external_facts(facts, [scope])), 1)

    def test_duplicate_disclosures_merge_across_event_ids_and_related_rules_combine(self):
        records = [self.record("e1", 1, "Acme export allows at most 4 pages."),
                   self.record("e2", 2, "Acme export allows at most 4 pages."),
                   self.record("e3", 3, "Acme export includes signed notes.")]
        duplicate, = self.scopes(records[:2], [self.event("a", "e1"), self.event("b", "e2")])
        self.assertEqual(duplicate["external_event_ids"], ["a", "b"])
        groups = self.scopes(records, [self.event("a", "e1", focus="Acme export"),
                                      self.event("b", "e3", focus="Acme export")])
        self.assertEqual(len(groups), 3)
        combined = groups[-1]
        self.assertEqual(combined["external_source_ids"], ["e1", "e3"])

    def test_static_complexity_counts_distinct_closed_facts_and_marks_unknown_type(self):
        records = [self.record("e1", 1, "Acme has a four page limit."),
                   self.record("e2", 2, "Acme signed notes remain.")]
        scope = self.scopes(records, [self.event("a", "e1", focus="Acme export"),
                                     self.event("b", "e2", focus="Acme export")])[-1]
        facts = [dict(id="f1", statement=records[0]["text"], sources=["e1"]),
                 dict(id="f2", statement=records[1]["text"], sources=["e2"])]
        index = build_evidence_index(facts, [scope], "memory")
        group = dict(qa_mode="memory", scope=scope, facts=facts)
        question = dict(answer_points=[dict(text=row["statement"], sources=row["sources"]) for row in facts])
        labels = static_candidate_labels(group, question, index, "M4")
        self.assertEqual(labels["difficulty"], "medium")
        self.assertEqual(labels["type"], "unknown")
        self.assertEqual(labels["type_candidates"], [])
        question["answer_points"][0]["text"] = "Correction exception: " + facts[0]["statement"]
        self.assertEqual(static_candidate_labels(group, question, index, "M4")["difficulty"], "hard")
        question["answer_points"][0]["sources"].append("missing")
        self.assertEqual(static_candidate_labels(group, question, index, "M4")["difficulty"], "unknown")
        balance = difficulty_balance([dict(qa_mode="memory", **labels)], {"memory": 10})["memory"]
        self.assertEqual(balance["target"], {"easy": 3, "medium": 4, "hard": 3})
        self.assertEqual(type_balance([], {"memory": 1})["memory"]["actual"],
                         {"M1": 0, "M2": 0, "M3": 0, "M4": 0, "M5": 0, "M6": 0})

    def test_co_disclosed_unused_fact_does_not_raise_complexity(self):
        records = [self.record("e1", 1, "Acme export allows four pages. Signed notes remain.")]
        scope, = self.scopes(records, [self.event("a", "e1")])
        facts = [dict(id="f1", statement="Acme export allows four pages.", sources=["e1"]),
                 dict(id="f2", statement="Signed notes remain.", sources=["e1"])]
        group = dict(qa_mode="memory", scope=scope, facts=facts)
        index = build_evidence_index(facts, [scope], "memory")
        candidate = dict(answer_points=[dict(text=facts[0]["statement"], sources=["e1"])])
        labels = static_candidate_labels(group, candidate, index, "M2")
        self.assertEqual(labels["difficulty"], "easy")
        self.assertEqual(labels["difficulty_fact_count"], 1)
        candidate["answer_points"][0]["text"] = "Apply the recorded rule."
        self.assertEqual(static_candidate_labels(group, candidate, index, "M2")["difficulty"], "unknown")

    def test_external_duplicate_obligation_is_independent_of_disclosure_ids(self):
        left = dict(qa_mode="memory", question="What rule applies to Acme export?",
                    answer_target="Acme export", answer_points=[dict(text="Limit four pages.", sources=["e1"])])
        right = dict(left, answer_points=[dict(text="Limit four pages.", sources=["e2"])])
        self.assertEqual(duplicate_reason(left, right), "near_duplicate_answer_target")

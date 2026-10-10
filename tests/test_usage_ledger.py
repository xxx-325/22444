import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dialogue_benchmark import llm
from dialogue_benchmark.collection import episode_usage
from dialogue_benchmark.task_eval.artifacts import save


def envelope(content="NO_FACTS", **usage):
    return io.BytesIO(json.dumps({"choices": [
        {"finish_reason": "stop", "message": {"content": content}}], "usage": usage}).encode())


class UsageLedgerTests(unittest.TestCase):
    def test_each_finished_request_is_appended_including_failures(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict("os.environ", {"KEY": "test-only"}), \
                patch("dialogue_benchmark.llm.urllib.request.build_opener") as opener:
            ledger = Path(directory) / "usage-ledger.jsonl"
            previous = llm.set_usage_ledger(ledger)
            try:
                client = llm.ChatClient("https://example.invalid", "model", "KEY")
            finally:
                llm.set_usage_ledger(previous)
            unbound = llm.ChatClient("https://example.invalid", "model", "KEY")
            opener.return_value.open.side_effect = [
                envelope(prompt_tokens=10, completion_tokens=2),
                envelope("not a protocol", prompt_tokens=7, completion_tokens=1),
                envelope(prompt_tokens=1, completion_tokens=1)]
            client.ask("p", {})
            with self.assertRaises(llm.ModelStageError):
                client.ask("p", {})
            unbound.ask("p", {})
            rows = llm.read_usage_ledger(ledger)
        self.assertEqual([(row["status"], row["prompt_tokens"]) for row in rows],
                         [("completed", 10), ("protocol_error", 7)])

    def test_episode_usage_counts_interrupted_requests_from_the_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "qa").mkdir()
            rows = [dict(request_count=1, prompt_tokens=10, completion_tokens=2, status="completed")] * 3
            (root / "qa/usage-ledger.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows) + '{"torn')
            # A receipt saved by an earlier, interrupted run must not be added again.
            save(root / "usage.json", {"receipts": [dict(path="qa/manifest.json", requests=0,
                                                         prompt_tokens=0, completion_tokens=0,
                                                         complete=False)]})
            interrupted = episode_usage(root)
            self.assertEqual((interrupted["requests"], interrupted["total_tokens"]), (3, 36))
            self.assertFalse(interrupted["complete"])
            # The resumed manifest lists only its own requests; the ledger stays authoritative.
            save(root / "qa/manifest.json", {"usage": [rows[0]]})
            finished = episode_usage(root)
            self.assertEqual((finished["requests"], finished["total_tokens"]), (3, 36))
            self.assertTrue(finished["complete"])
            self.assertEqual([row["path"] for row in finished["receipts"]], ["qa/usage-ledger.jsonl"])


if __name__ == "__main__":
    unittest.main()

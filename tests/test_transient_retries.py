import unittest
from unittest.mock import Mock

from dialogue_benchmark.llm import ModelStageError, retry_model_call


class TransientRetryTests(unittest.TestCase):
    def test_transport_failure_reuses_same_call_and_stops_after_success(self):
        call = Mock(side_effect=[
            ModelStageError("http_error", http_status=429),
            ModelStageError("timeout"),
            {"questions": []},
        ])
        sleep = Mock()
        self.assertEqual(retry_model_call(call, sleep=sleep), {"questions": []})
        self.assertEqual(call.call_count, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [0.25, 0.5])

    def test_retry_exhaustion_preserves_failure(self):
        error = ModelStageError("http_error", http_status=503)
        call = Mock(side_effect=error)
        with self.assertRaises(ModelStageError) as caught:
            retry_model_call(call, sleep=Mock())
        self.assertIs(caught.exception, error)
        self.assertEqual(call.call_count, 3)

    def test_deterministic_failures_are_not_retried(self):
        for error in (
            ModelStageError("http_error", http_status=401),
            ModelStageError("http_error", http_status=402),
            ModelStageError("protocol_error"),
            ModelStageError("request_budget"),
            ModelStageError("credential_detected"),
            ValueError("source closure"),
        ):
            with self.subTest(error=error):
                call, sleep = Mock(side_effect=error), Mock()
                with self.assertRaises(type(error)):
                    retry_model_call(call, sleep=sleep)
                call.assert_called_once()
                sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()

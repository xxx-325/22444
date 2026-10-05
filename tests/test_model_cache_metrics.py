import unittest

from dialogue_benchmark.task_eval.metrics import cache_usage


class ModelCacheMetricsTest(unittest.TestCase):
    def test_deepseek_hit_and_miss(self):
        result = cache_usage([{"prompt_cache_hit_tokens": 8, "prompt_cache_miss_tokens": 2}])
        self.assertEqual(result["cache_hit_tokens"], 8)
        self.assertEqual(result["cache_miss_tokens"], 2)
        self.assertEqual(result["cache_observed_prompt_tokens"], 10)
        self.assertTrue(result["cache_usage_complete"])
        self.assertEqual(result["cache_hit_rate"], 0.8)

    def test_openai_cached_tokens_derives_miss(self):
        result = cache_usage([{"prompt_tokens": 10,
                               "prompt_tokens_details": {"cached_tokens": 4}}])
        self.assertEqual((result["cache_hit_tokens"], result["cache_miss_tokens"]), (4, 6))

    def test_missing_and_zero_are_distinct(self):
        self.assertFalse(cache_usage([{"prompt_tokens": 10}])["cache_usage_complete"])
        result = cache_usage([{"prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0}])
        self.assertTrue(result["cache_usage_complete"])
        self.assertIsNone(result["cache_hit_rate"])

    def test_deepseek_fields_win_without_double_counting(self):
        result = cache_usage([{"prompt_cache_hit_tokens": 3, "prompt_cache_miss_tokens": 7,
                               "prompt_tokens": 10,
                               "prompt_tokens_details": {"cached_tokens": 3}}])
        self.assertEqual(result["cache_hit_tokens"], 3)
        self.assertEqual(result["cache_miss_tokens"], 7)


if __name__ == "__main__":
    unittest.main()

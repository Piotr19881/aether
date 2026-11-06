import unittest

from src.ai_view import TokenizerProfile
from src.dev_benchmark import measure


class ByteEncoder:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class DevBenchmarkTests(unittest.TestCase):
    def test_measure_reports_latency_memory_and_input_size(self) -> None:
        profile = TokenizerProfile(
            name="test",
            backend="tiktoken",
            tokenizer="unused",
            operator_aliases={},
            idiom_aliases={},
            semantic_aliases={},
            alias_candidates=("$1",),
        )
        result = measure(
            "fn identity(value: i32) { return value; }\n",
            iterations=3,
            profile=profile,
            encoder=ByteEncoder(),
        )

        self.assertEqual(result["iterations"], 3)
        self.assertEqual(result["source_bytes"], 42)
        self.assertGreaterEqual(result["median_ms"], 0)
        self.assertGreaterEqual(result["p95_ms"], 0)
        self.assertGreater(result["peak_kib"], 0)

    def test_invalid_input_is_reported_instead_of_benchmarked(self) -> None:
        with self.assertRaisesRegex(ValueError, "E_SYNTAX"):
            measure("not a declaration", iterations=1, encoder=ByteEncoder())


if __name__ == "__main__":
    unittest.main()

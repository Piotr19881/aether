from pathlib import Path
import unittest

from src.parser import ParseError, parse, parse_file, render


class ParserTests(unittest.TestCase):
    def test_lossless_roundtrip_for_sample_file(self) -> None:
        path = Path(__file__).parents[1] / "examples" / "sensor.hl"
        with path.open("r", encoding="utf-8", newline="") as source_file:
            source = source_file.read()

        self.assertEqual(render(parse_file(path)), source)

    def test_preserves_comments_whitespace_and_pipeline_tokens(self) -> None:
        source = '# header\nfn f(x: V) -> V {\n  return x |> scale(2.0) // inline\n}\n'
        tree = parse(source)

        self.assertEqual(render(tree), source)

        def all_nodes(node):
            yield node
            for child in node.children:
                yield from all_nodes(child)

        nodes = list(all_nodes(tree))
        self.assertTrue(any(node.kind == "COMMENT" for node in nodes))
        self.assertTrue(any(node.kind == "WHITESPACE" for node in nodes))
        self.assertTrue(any(node.value == "|>" for node in nodes))
        self.assertTrue(any(node.kind == "group_brace" for node in nodes))

    def test_reports_mismatched_delimiter_with_location(self) -> None:
        with self.assertRaisesRegex(ParseError, "line 1, column 3"):
            parse("([)")

    def test_reports_unterminated_string(self) -> None:
        with self.assertRaisesRegex(ParseError, "Unterminated string literal"):
            parse('let value = "unfinished')


if __name__ == "__main__":
    unittest.main()

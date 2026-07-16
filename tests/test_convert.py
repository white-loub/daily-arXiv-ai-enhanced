import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTER = REPO_ROOT / "to_md" / "convert.py"


def make_paper(paper_id, title, ai=None):
    paper = {
        "id": paper_id,
        "title": title,
        "authors": ["Author"],
        "summary": "Summary",
        "abs": "https://example.com/paper",
        "categories": ["cs.CL"],
    }
    if ai is not None:
        paper["AI"] = ai
    return paper


COMPLETE_AI = {
    "tldr": "TLDR",
    "motivation": "Motivation",
    "method": "Method",
    "result": "Result",
    "conclusion": "Conclusion",
}


class ConvertTests(unittest.TestCase):
    def run_converter(self, papers):
        temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(temp_dir.cleanup)
        data_path = Path(temp_dir.name) / "sample_AI_enhanced_Chinese.jsonl"
        data_path.write_text(
            "".join(json.dumps(paper) + "\n" for paper in papers)
        )
        result = subprocess.run(
            [sys.executable, str(CONVERTER), "--data", str(data_path)],
            cwd=CONVERTER.parent,
            capture_output=True,
            text=True,
        )
        return result, data_path.with_name("sample.md")

    def test_incomplete_ai_is_skipped_before_totals_are_counted(self):
        result, output_path = self.run_converter([
            make_paper("valid", "Valid paper", COMPLETE_AI),
            make_paper("invalid", "Invalid paper"),
        ])

        self.assertEqual(result.returncode, 0, result.stderr)
        markdown = output_path.read_text()
        self.assertIn("[Total: 1]", markdown)
        self.assertIn("Valid paper", markdown)
        self.assertNotIn("Invalid paper", markdown)

    def test_all_incomplete_ai_fails_without_writing_markdown(self):
        result, output_path = self.run_converter([
            make_paper("invalid", "Invalid paper"),
        ])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No papers contain complete AI enhancement data", result.stderr)
        self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()

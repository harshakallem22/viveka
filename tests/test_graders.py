"""Tests for numeric extraction and deterministic grading.

Most cases here are real failures observed on benchmark output, not invented ones. Extraction is
the single easiest place to ship a silently wrong leaderboard, because a bad rule looks exactly
like a bad model.
"""

from __future__ import annotations

import pytest

from viveka.core.graders import (
    GRADERS,
    NumericExactMatch,
    extract_final_number,
    get_grader,
    parse_number,
)
from viveka.core.types import GoldenItem


def item(reference: str = "18") -> GoldenItem:
    return GoldenItem(
        id="gsm8k-test-00013",
        source="gsm8k",
        category="math_reasoning",
        question="how many?",
        reference_answer=reference,
        grader="numeric_exact_match",
    )


class TestParseNumber:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("18", 18.0),
            ("1,430", 1430.0),      # ~1% of GSM8K answers carry thousands separators
            ("34.00", 34.0),        # mis-scored 10% of a real smoke sample as a string compare
            ("-7", -7.0),           # 2 of 1319 GSM8K test answers are negative
            ("$8000", 8000.0),
            ("50%", 50.0),
            ("18.", 18.0),          # trailing sentence period
            (" 42 ", 42.0),
        ],
    )
    def test_normalizes(self, raw, expected):
        assert parse_number(raw) == expected

    @pytest.mark.parametrize("raw", ["", "  ", "-", ".", "abc", "N/A"])
    def test_rejects_non_numbers(self, raw):
        assert parse_number(raw) is None


class TestExtractFinalNumber:
    def test_prefers_the_marker_our_prompt_asks_for(self):
        text = "I computed 16 - 3 - 4 = 9 and then 9 * 2 = 18.\n#### 18"
        assert extract_final_number(text) == 18.0

    def test_last_marker_wins_when_a_model_restates(self):
        assert extract_final_number("#### 18\nOn reflection:\n#### 20") == 20.0

    def test_handles_boxed_latex(self):
        assert extract_final_number(r"therefore $\boxed{72}$") == 72.0

    def test_handles_bold_markdown(self):
        assert extract_final_number("Total water removed = 3 + 6 + 20 = **29 liters**") == 29.0

    def test_falls_back_to_last_number_on_the_last_numeric_line(self):
        text = "Step 1: 5 * 2 = 10\nStep 2: add two\nThe final answer is 12\n\n"
        assert extract_final_number(text) == 12.0

    def test_scans_from_the_end_past_non_numeric_trailing_lines(self):
        text = "The answer is 145\nLet me know if you need more detail!"
        assert extract_final_number(text) == 145.0

    def test_none_when_no_number_present(self):
        assert extract_final_number("I am not sure how to solve this.") is None

    def test_none_on_empty(self):
        assert extract_final_number("") is None

    def test_loose_fallback_can_be_disabled(self):
        """`allow_loose=False` is how truncated responses avoid a meaningless guess."""
        text = "4. Their mother then made 4 pies"
        assert extract_final_number(text) == 4.0
        assert extract_final_number(text, allow_loose=False) is None

    def test_marker_still_honoured_when_loose_is_disabled(self):
        assert extract_final_number("blah\n#### 18", allow_loose=False) == 18.0


class TestNumericExactMatch:
    def setup_method(self):
        self.grader = NumericExactMatch()

    def test_correct(self):
        s = self.grader.grade(item("18"), "reasoning\n#### 18")
        assert s.verdict == "correct" and s.score == 1.0

    def test_incorrect(self):
        s = self.grader.grade(item("18"), "reasoning\n#### 21")
        assert s.verdict == "incorrect" and s.score == 0.0

    def test_decimal_and_integer_forms_agree(self):
        """The D24 fix: 34.00 == 34. A string comparison mis-scored this in a real sample."""
        assert self.grader.grade(item("34"), "#### 34.00").verdict == "correct"
        assert self.grader.grade(item("34"), "#### 34.0").verdict == "correct"

    def test_thousands_separators_agree(self):
        assert self.grader.grade(item("1430"), "#### 1,430").verdict == "correct"

    def test_negative_answers(self):
        assert self.grader.grade(item("-7"), "#### -7").verdict == "correct"
        assert self.grader.grade(item("-7"), "#### 7").verdict == "incorrect"

    def test_unextractable_is_error_not_incorrect(self):
        """The distinction that keeps a broken regex from looking like a bad model (§8.1)."""
        s = self.grader.grade(item("18"), "I cannot solve this problem.")
        assert s.verdict == "error"
        assert s.error == "unextractable"
        assert not s.to_judgement(1, grader_name="numeric_exact_match").is_scorable

    def test_broken_reference_blames_the_data_not_the_model(self):
        s = self.grader.grade(item("not-a-number"), "#### 18")
        assert s.verdict == "error"
        assert "reference answer" in (s.error or "")

    def test_truncated_without_a_marker_is_incorrect_not_a_lucky_guess(self):
        """Regression guard for a real near-miss.

        A cut-off response ending "4. Their mother then made 4 pies" yields 4 from a *list
        marker*. If the reference happened to be 4, loose extraction would score it correct and
        silently inflate the leaderboard. A model that never emitted an answer must be incorrect.
        """
        text = "Let me work through this.\n4. Their mother then made 4 pies"
        assert self.grader.grade(item("4"), text).verdict == "correct"  # loose path, unflagged
        truncated = self.grader.grade(item("4"), text, truncated=True)
        assert truncated.verdict == "incorrect"
        assert "token cap" in (truncated.reasoning or "")

    def test_truncated_with_a_marker_is_still_scored(self):
        """If the model did emit the marker before running out, trust it."""
        s = self.grader.grade(item("18"), "reasoning\n#### 18\nand furthermore", truncated=True)
        assert s.verdict == "correct"

    def test_reasoning_explains_the_comparison(self):
        s = self.grader.grade(item("18"), "#### 21")
        assert "21" in (s.reasoning or "") and "18" in (s.reasoning or "")

    def test_never_raises_on_hostile_input(self):
        for text in ["", "   ", "\n\n", "####", "#### abc", "$", "-", "." * 500]:
            assert self.grader.grade(item("18"), text).verdict in {
                "correct",
                "incorrect",
                "error",
            }


class TestRegistry:
    def test_numeric_grader_is_registered(self):
        assert get_grader("numeric_exact_match").name == "numeric_exact_match"

    def test_unknown_grader_names_what_is_available(self):
        with pytest.raises(KeyError, match="numeric_exact_match"):
            get_grader("does_not_exist")

    def test_llm_judge_is_not_in_the_default_registry(self):
        """It needs a Provider, and `core.graders` must stay free of network concerns."""
        assert "llm_judge" not in GRADERS

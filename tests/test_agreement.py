"""Tests for judge-reliability statistics.

Kappa values here are hand-computed from the 2x2 formula, not captured from the implementation.
The critical property is the one in `test_always_correct_judge_scores_zero_kappa`: it is the whole
reason we report kappa instead of raw agreement.
"""

from __future__ import annotations

import pytest

from viveka.core.agreement import cohen_kappa, compare_verdicts


class TestCohenKappa:
    def test_perfect_agreement(self):
        assert cohen_kappa(tp=25, fp=0, fn=0, tn=25) == pytest.approx(1.0)

    def test_hand_computed_value(self):
        """tp=40 fp=10 fn=5 tn=45, n=100.

        p_o = 85/100 = 0.85
        judge_correct = 50/100 = 0.5 ; truth_correct = 45/100 = 0.45
        p_e = 0.5*0.45 + 0.5*0.55 = 0.225 + 0.275 = 0.50
        kappa = (0.85 - 0.50) / 0.50 = 0.70
        """
        assert cohen_kappa(tp=40, fp=10, fn=5, tn=45) == pytest.approx(0.70, abs=1e-9)

    def test_chance_level_agreement_is_zero(self):
        # tp=25 fp=25 fn=25 tn=25: p_o=0.5, p_e=0.5 -> kappa 0
        assert cohen_kappa(tp=25, fp=25, fn=25, tn=25) == pytest.approx(0.0)

    def test_worse_than_chance_is_negative(self):
        assert cohen_kappa(tp=5, fp=45, fn=45, tn=5) < 0

    def test_empty_table(self):
        assert cohen_kappa(0, 0, 0, 0) == 0.0

    def test_degenerate_single_class_agreeing(self):
        """All truth-correct and judge agrees: expected agreement is already total, so kappa is
        undefined. Returning 1.0 for genuine total agreement is defensible; claiming credit for
        partial agreement in that regime would not be."""
        assert cohen_kappa(tp=50, fp=0, fn=0, tn=0) == 1.0
        assert cohen_kappa(tp=45, fp=0, fn=5, tn=0) == 0.0


class TestCompareVerdicts:
    def test_confusion_matrix_orientation(self):
        pairs = [
            ("correct", "correct"),      # tp
            ("incorrect", "correct"),    # fp -- judge approved a wrong answer
            ("correct", "incorrect"),    # fn
            ("incorrect", "incorrect"),  # tn
        ]
        r = compare_verdicts(pairs)
        assert (r.true_positive, r.false_positive, r.false_negative, r.true_negative) == (1, 1, 1, 1)
        assert r.n == 4
        assert r.raw_agreement == 0.5

    def test_error_verdicts_are_skipped_not_counted_as_disagreement(self):
        """An unextractable answer is an absence of measurement, not a judge mistake."""
        pairs = [
            ("correct", "correct"),
            ("error", "correct"),
            ("correct", "error"),
        ]
        r = compare_verdicts(pairs)
        assert r.n == 1
        assert r.n_skipped == 2
        assert r.raw_agreement == 1.0

    def test_always_correct_judge_gets_high_agreement_but_zero_kappa(self):
        """The single most important test in this file.

        A judge that blindly says "correct" scores 90% raw agreement on a set that is 90%
        correct -- while being completely useless. Kappa exposes it. This is why the CLI quotes
        kappa as the headline and warns when class balance is skewed.
        """
        pairs = [("correct", "correct")] * 90 + [("incorrect", "correct")] * 10
        r = compare_verdicts(pairs)
        assert r.raw_agreement == pytest.approx(0.90)
        assert r.cohen_kappa == pytest.approx(0.0)
        assert r.interpretation == "no better than chance"
        assert r.false_positive_rate == 1.0  # it approved every wrong answer

    def test_false_positive_rate_is_relative_to_wrong_answers(self):
        pairs = (
            [("incorrect", "correct")] * 3      # 3 wrongly approved
            + [("incorrect", "incorrect")] * 7  # 7 correctly rejected
            + [("correct", "correct")] * 10
        )
        r = compare_verdicts(pairs)
        assert r.false_positive == 3
        assert r.false_positive_rate == pytest.approx(0.3)  # 3 of 10 wrong answers

    def test_false_negative_rate_is_relative_to_right_answers(self):
        pairs = [("correct", "incorrect")] * 2 + [("correct", "correct")] * 8
        r = compare_verdicts(pairs)
        assert r.false_negative == 2
        assert r.false_negative_rate == pytest.approx(0.2)
        assert r.false_positive_rate is None  # no wrong answers to approve

    def test_truth_positive_rate_flags_class_imbalance(self):
        pairs = [("correct", "correct")] * 95 + [("incorrect", "incorrect")] * 5
        assert compare_verdicts(pairs).truth_positive_rate == pytest.approx(0.95)

    def test_empty_input_is_safe(self):
        r = compare_verdicts([])
        assert r.n == 0
        assert r.cohen_kappa == 0.0
        assert r.interpretation == "no comparable pairs"
        assert r.false_positive_rate is None

    @pytest.mark.parametrize(
        ("kappa_pairs", "expected"),
        [
            (([("correct", "correct")] * 50 + [("incorrect", "incorrect")] * 50), "almost perfect"),
            (
                (
                    [("correct", "correct")] * 40
                    + [("incorrect", "correct")] * 10
                    + [("correct", "incorrect")] * 5
                    + [("incorrect", "incorrect")] * 45
                ),
                "substantial",
            ),
        ],
    )
    def test_interpretation_labels(self, kappa_pairs, expected):
        assert compare_verdicts(kappa_pairs).interpretation == expected

    def test_as_dict_includes_the_interpretation(self):
        import json

        d = compare_verdicts([("correct", "correct"), ("incorrect", "incorrect")]).as_dict()
        assert d["interpretation"] == "almost perfect"
        assert json.loads(json.dumps(d))["cohen_kappa"] == 1.0

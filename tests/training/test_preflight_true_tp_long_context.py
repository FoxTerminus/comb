import copy
import math
import unittest

from training.preflight_true_tp_long_context import capacity_checks


def memory() -> dict[str, int]:
    return {
        "allocated_bytes": 40,
        "reserved_bytes": 60,
        "peak_allocated_bytes": 50,
        "peak_reserved_bytes": 70,
        "total_bytes": 100,
    }


def rank_report(rank: int, loss: float = 0.5) -> dict[str, object]:
    report: dict[str, object] = {
        "rank": rank,
        "loss": loss,
        "context_tokens": 16_255,
        "context_nonpadding_tokens": 9_095,
        "query_tokens": 512,
        "engine_global_steps_before": 81_000,
        "engine_global_steps_after": 81_000,
        "optimizer_state_step_advanced": True,
        "parameter_update_max_abs_delta": 0.001,
        "parameter_update_observed": True,
    }
    for phase in ("baseline", "after_forward", "after_backward", "after_optimizer"):
        report[phase] = memory()
    return report


class CapacityChecksTest(unittest.TestCase):
    def check(self, ranks: list[dict[str, object]]) -> dict[str, object]:
        return capacity_checks(
            ranks,
            context_tokens=16_255,
            context_nonpadding_tokens=9_095,
            query_tokens=512,
        )

    def test_accepts_consistent_finite_ranks_with_memory_headroom(self) -> None:
        result = self.check([rank_report(0), rank_report(1)])
        self.assertTrue(result["passed"])
        self.assertEqual(result["minimum_peak_reserved_headroom_bytes"], 30)

    def test_rejects_nonfinite_or_divergent_loss(self) -> None:
        self.assertFalse(self.check([rank_report(0), rank_report(1, math.inf)])["passed"])
        self.assertFalse(self.check([rank_report(0), rank_report(1, 0.51)])["passed"])

    def test_rejects_shape_mismatch(self) -> None:
        ranks = [rank_report(0), rank_report(1)]
        ranks[1]["context_nonpadding_tokens"] = 9_094
        self.assertFalse(self.check(ranks)["passed"])

    def test_rejects_invalid_memory_accounting(self) -> None:
        ranks = [rank_report(0), rank_report(1)]
        broken = copy.deepcopy(ranks)
        broken[1]["after_backward"]["peak_reserved_bytes"] = 101
        result = self.check(broken)
        self.assertFalse(result["cuda_memory_accounting_valid"])
        self.assertFalse(result["passed"])

    def test_rejects_missing_optimizer_update(self) -> None:
        ranks = [rank_report(0), rank_report(1)]
        ranks[1]["optimizer_state_step_advanced"] = False
        result = self.check(ranks)
        self.assertFalse(result["optimizer_update_observed_all_ranks"])
        self.assertFalse(result["passed"])

    def test_rejects_missing_parameter_update_or_global_step_change(self) -> None:
        ranks = [rank_report(0), rank_report(1)]
        ranks[0]["parameter_update_observed"] = False
        self.assertFalse(self.check(ranks)["passed"])
        ranks = [rank_report(0), rank_report(1)]
        ranks[0]["engine_global_steps_after"] = 81_001
        self.assertFalse(self.check(ranks)["passed"])

    def test_requires_rank_reports(self) -> None:
        with self.assertRaises(ValueError):
            self.check([])


if __name__ == "__main__":
    unittest.main()

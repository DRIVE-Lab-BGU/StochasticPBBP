from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.Rollout import RolloutTrace  # noqa: E402
from utils.R2Noise import R2GradientAdditiveNoise  # noqa: E402


class R2NoiseTest(unittest.TestCase):
    @staticmethod
    def _build_trace_with_two_steps():
        step0_a = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        step0_b = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        step1_a = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
        step1_b = torch.tensor([-1.0], dtype=torch.float64, requires_grad=True)

        objective = (
            step0_a.pow(2).sum()
            + (2.0 * step0_b).sum()
            + 0.5 * step0_b.pow(2).sum()
            + 0.25 * step1_a.pow(2).sum()
            + 3.0 * step1_b.pow(2).sum()
        )
        trace = RolloutTrace(
            observations=[{}, {}],
            actions=[
                {'a': step0_a, 'b': step0_b},
                {'a': step1_a, 'b': step1_b},
            ],
            rewards=[],
            terminals=[],
            final_observation={},
            final_subs={},
            policy_state=None,
            model_params={},
        )
        return trace, objective

    @staticmethod
    def _build_trace_with_vector_action():
        action = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
        objective = action[0].pow(3) + 0.5 * action[1].pow(2)
        trace = RolloutTrace(
            observations=[{}],
            actions=[{'a': action}],
            rewards=[],
            terminals=[],
            final_observation={},
            final_subs={},
            policy_state=None,
            model_params={},
        )
        return trace, objective

    def test_refresh_builds_gradient_only_quantile_normalized_timestep_profile(self) -> None:
        trace, objective = self._build_trace_with_two_steps()
        noise = R2GradientAdditiveNoise(
            min_std=0.1,
            max_std=1.0,
            alpha=1.0,
            step_score_aggregate='mean',
            normalization_quantile=0.95,
        )

        profile = noise.refresh_from_analysis_trace(trace=trace, objective=objective)

        score_step0 = (2.0 + 4.0) / 2.0
        score_step1 = (0.25 + 6.0) / 2.0
        expected_quantile = float(
            torch.quantile(
                torch.tensor([score_step0, score_step1], dtype=torch.float64),
                0.95,
            ).item()
        )
        expected_sigma0 = 0.1 + (1.0 - 0.1) * (1.0 - min(score_step0 / expected_quantile, 1.0))
        expected_sigma1 = 0.1

        self.assertEqual(len(profile), 2)
        self.assertTrue(torch.allclose(profile[0]['a'], torch.full_like(profile[0]['a'], expected_sigma0)))
        self.assertTrue(torch.allclose(profile[0]['b'], torch.full_like(profile[0]['b'], expected_sigma0)))
        self.assertTrue(torch.allclose(profile[1]['a'], torch.full_like(profile[1]['a'], expected_sigma1)))
        self.assertTrue(torch.allclose(profile[1]['b'], torch.full_like(profile[1]['b'], expected_sigma1)))
        self.assertAlmostEqual(noise.last_action_scores[0]['a'], 2.0, places=6)
        self.assertAlmostEqual(noise.last_action_scores[0]['b'], 4.0, places=6)
        self.assertAlmostEqual(noise.last_action_scores[1]['a'], 0.25, places=6)
        self.assertAlmostEqual(noise.last_action_scores[1]['b'], 6.0, places=6)
        self.assertAlmostEqual(noise.last_curvatures[0]['a'], 0.0, places=6)
        self.assertAlmostEqual(noise.last_curvatures[1]['b'], 0.0, places=6)
        self.assertAlmostEqual(noise.last_step_scores[0], score_step0, places=6)
        self.assertAlmostEqual(noise.last_step_scores[1], score_step1, places=6)
        self.assertAlmostEqual(noise.last_score_quantile, expected_quantile, places=6)

    def test_action_score_is_gradient_norm_only(self) -> None:
        trace, objective = self._build_trace_with_vector_action()
        noise = R2GradientAdditiveNoise(
            min_std=0.0,
            max_std=1.0,
            alpha=1.0,
            normalization_quantile=1.0,
        )

        noise.refresh_from_analysis_trace(trace=trace, objective=objective)

        expected_gradient_norm = math.sqrt(13.0)
        self.assertAlmostEqual(noise.last_action_scores[0]['a'], expected_gradient_norm, places=6)
        self.assertAlmostEqual(noise.last_curvatures[0]['a'], 0.0, places=6)

    def test_curvature_compatibility_fields_do_not_change_scores(self) -> None:
        base_trace, base_objective = self._build_trace_with_vector_action()
        compatibility_trace, compatibility_objective = self._build_trace_with_vector_action()
        base_noise = R2GradientAdditiveNoise(
            min_std=0.0,
            max_std=1.0,
            alpha=1.0,
            normalization_quantile=1.0,
        )
        compatibility_noise = R2GradientAdditiveNoise(
            min_std=0.0,
            max_std=1.0,
            alpha=1.0,
            curvature_weight=10.0,
            curvature_reduce='norm',
            normalization_quantile=1.0,
        )

        base_noise.refresh_from_analysis_trace(trace=base_trace, objective=base_objective)
        compatibility_noise.refresh_from_analysis_trace(
            trace=compatibility_trace,
            objective=compatibility_objective,
        )

        self.assertAlmostEqual(
            compatibility_noise.last_action_scores[0]['a'],
            base_noise.last_action_scores[0]['a'],
            places=6,
        )
        self.assertAlmostEqual(compatibility_noise.last_curvatures[0]['a'], 0.0, places=6)

    def test_step_score_aggregate_sum_adds_gradient_norms(self) -> None:
        trace, objective = self._build_trace_with_two_steps()
        noise = R2GradientAdditiveNoise(
            min_std=0.0,
            max_std=1.0,
            alpha=1.0,
            step_score_aggregate='sum',
            normalization_quantile=1.0,
        )

        noise.refresh_from_analysis_trace(trace=trace, objective=objective)

        expected_step0 = 2.0 + 4.0
        expected_step1 = 0.25 + 6.0

        self.assertAlmostEqual(noise.last_step_scores[0], expected_step0, places=6)
        self.assertAlmostEqual(noise.last_step_scores[1], expected_step1, places=6)


if __name__ == '__main__':
    unittest.main()

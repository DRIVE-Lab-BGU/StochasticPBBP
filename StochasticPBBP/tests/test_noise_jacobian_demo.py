from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pyRDDLGym
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils.Noise import noise  # noqa: E402


DOMAIN = PACKAGE_ROOT / "problems" / "reservoir" / "domain.rddl"
INSTANCE = PACKAGE_ROOT / "problems" / "reservoir" / "instance_1.rddl"


class NoiseJacobianDemoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        env = pyRDDLGym.make(domain=DOMAIN, instance=INSTANCE, vectorized=True)
        cls.model = env.model

    @staticmethod
    def _make_demo_point(reference: torch.Tensor, start_value: float) -> torch.Tensor:
        values = torch.arange(
            start_value,
            start_value + float(reference.numel()),
            dtype=reference.dtype,
            device=reference.device,
        )
        return values.reshape(reference.shape)

    def test_print_point_jacobian_and_norm(self) -> None:
        noise_helper = noise(
            action_dim=1,
            max_action=1.0,
            horizon=self.model.horizon,
            model=self.model,
        )
        compiler = noise_helper._get_compiler()

        point = {
            'rlevel': self._make_demo_point(compiler.init_values['rlevel'], start_value=1.0),
            'release': self._make_demo_point(compiler.init_values['release'], start_value=10.0),
        }

        jacobians = noise_helper.cpf_jacobian(
            cpf_name="rlevel'",
            subs=point,
            wrt=['rlevel', 'release'],
        )
        jacobian_norm = noise_helper.cpf_jacobian_norm(
            cpf_name="rlevel'",
            subs=point,
            wrt=['rlevel', 'release'],
        )

        print('\n=== Point (subs) ===')
        for (name, value) in point.items():
            print(f'{name} = {value}')

        print('\n=== Jacobian of rlevel\' ===')
        for (name, value) in jacobians.items():
            print(f"d(rlevel')/d({name}) shape={tuple(value.shape)}")
            print(value)

        print('\n=== Jacobian Norm ===')
        print(jacobian_norm)

        self.assertIn('rlevel', jacobians)
        self.assertIn('release', jacobians)
        output_shape = jacobians['rlevel'].shape[:point['rlevel'].ndim]
        self.assertEqual(jacobians['rlevel'].shape, output_shape + point['rlevel'].shape)
        self.assertEqual(jacobians['release'].shape, output_shape + point['release'].shape)
        self.assertEqual(jacobian_norm.ndim, 0)
        self.assertGreaterEqual(float(jacobian_norm.detach()), 0.0)


if __name__ == '__main__':
    unittest.main()

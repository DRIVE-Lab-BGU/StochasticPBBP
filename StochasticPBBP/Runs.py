from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyRDDLGym
import torch
from torch import nn
PACKAGE_ROOT = Path(__file__).resolve().parent
print(f"PACKAGE_ROOT={PACKAGE_ROOT}")
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.Logic import FuzzyLogic ,ExactLogic
from core.Rollout import TorchRollout
from core.Simulator import TorchRDDLSimulator



TensorDict = Dict[str, torch.Tensor]


def _as_tensor(value: Any,
               *,
               dtype: Optional[torch.dtype]=None,
               device: Optional[torch.device]=None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None or device is not None:
        tensor = tensor.to(dtype=dtype or tensor.dtype, device=device or tensor.device)
    return tensor



class GaussianPolicy(nn.Module):
    """State-independent diagonal Gaussian policy over the action vector."""

    def __init__(self,
                 action_template: TensorDict,
                 init_std: float=1.0,
                 min_log_std: float=-5.0,
                 max_log_std: float=2.0) -> None:
        super().__init__()
        if not action_template:
            raise ValueError('action_template must contain at least one tensor.')

        first_action = next(iter(action_template.values()))
        self.device = first_action.device
        self.dtype = first_action.dtype if first_action.dtype.is_floating_point else torch.float32
        self.action_specs = self._build_action_specs(action_template)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        output_dim = sum(spec['numel'] for spec in self.action_specs)
        init_log_std = float(math.log(max(init_std, 1e-6)))
        self.mu = nn.Parameter(
            torch.zeros(output_dim, device=self.device, dtype=self.dtype)
        )
        self.log_std = nn.Parameter(
            torch.full((output_dim,), init_log_std, device=self.device, dtype=self.dtype)
        )
    
    @staticmethod
    def _build_action_specs(action_template: TensorDict) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        for (name, template) in action_template.items():
            template_tensor = _as_tensor(template)
            if not template_tensor.dtype.is_floating_point:
                raise ValueError(
                    f'GaussianPolicy supports only real-valued action tensors, got {name} '
                    f'with dtype {template_tensor.dtype}.'
                )
            specs.append({
                'name': name,
                'shape': tuple(template_tensor.shape),
                'numel': int(template_tensor.numel()),
                'dtype': template_tensor.dtype,
                'device': template_tensor.device,
            })
        return specs

    def distribution(self) -> torch.distributions.Normal:
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        std = torch.exp(log_std).to(dtype=self.mu.dtype, device=self.mu.device)
        return torch.distributions.Normal(self.mu, std)

    def _pack_actions(self, flat_action: torch.Tensor) -> TensorDict:
        actions: TensorDict = {}
        start = 0
        for spec in self.action_specs:
            end = start + spec['numel']
            raw_action = flat_action[start:end].reshape(spec['shape'])
            actions[spec['name']] = raw_action.to(dtype=spec['dtype'], device=spec['device'])
            start = end
        return actions

    def forward(self,
                observation: TensorDict,
                step: Optional[int]=None,
                policy_state: Any=None,
                deterministic: bool=False) -> TensorDict:
        del observation, step, policy_state
        dist = self.distribution()
        # rsample uses the reparameterization trick so gradients reach mu/log_std.
        flat_action =  dist.rsample()
        return self._pack_actions(flat_action)
        
    
    
class Train:
    def __init__(self,
                 model: Any,
                 action_space: Optional[Any]=None,
                 policy: Optional[nn.Module]=None,
                 horizon: Optional[int]=None,
                 lr: float=1e-2,
                 hidden_sizes: Sequence[int]=(64, 64),
                 seed: int=0,
                 simulator: Optional[Any]=None) -> None:
        torch.manual_seed(seed)
        self.rollout = TorchRollout(model, horizon=horizon, logic=FuzzyLogic())
        self.rollout.cell.key.manual_seed(seed)
        self.simulator = simulator
        self.rollout.reset()
        if policy is None:
            policy = GaussianPolicy(
                action_template=self.rollout.noop_actions,
                
            )
        self.policy = policy
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=lr)

    def train_trajectory(self,
                         iterations: int=10,
                         print_every: int=1) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        self.policy.train()

        for iteration in range(1, iterations + 1):
            self.optimizer.zero_grad(set_to_none=True)
            trace = self.rollout(policy=self.policy)
            objective = trace.return_
            if objective.ndim > 0:
                objective = objective.mean()
            loss = -objective

            loss.backward()
            # print("###############################")
            # print("mu grad:", self.policy.mu.grad)
            # print("log_std grad:", self.policy.log_std.grad)
            # print("###############################")
            self.optimizer.step()

            metrics = {
                'iteration': float(iteration),
                'return': float(objective.detach()),
                'loss': float(loss.detach()),
                'steps': float(len(trace.rewards)),
            }
            history.append(metrics)

            if print_every > 0 and (
                iteration == 1 or iteration % print_every == 0 or iteration == iterations
            ):
                print(
                    f"iter={iteration:4d} "
                    f"return={metrics['return']:10.4f} "
                    f"loss={metrics['loss']:10.4f} "
                    f"steps={int(metrics['steps'])}"
                )
        if isinstance(self.policy, GaussianPolicy):
            dist = self.policy.distribution()
            print(f'self.policy.mu: {dist.mean.detach()}')
            print(f'self.policy.std: {dist.stddev.detach()}')
        return history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train a torch policy through differentiable rollouts.')
    parser.add_argument('--domain', type=Path, default=None, help='Path to the RDDL domain file.')
    parser.add_argument('--instance', type=Path, default=None, help='Path to the RDDL instance file.')
    parser.add_argument('--iterations', type=int, default=20, help='Number of gradient updates.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Adam learning rate.')
    parser.add_argument('--seed', type=int, default=0, help='Torch and rollout RNG seed.')
    parser.add_argument(
        '--hidden-sizes',
        type=int,
        nargs='*',
        default=[12, 12],
        help='Hidden layer sizes for the MLP policy.',
    )
    parser.add_argument(
        '--print-every',
        type=int,
        default=5,
        help='How often to print training metrics.',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()


    DOMAIN = PACKAGE_ROOT / "problems" / "reservoir" / "domain.rddl"
    INSTANCE = PACKAGE_ROOT / "problems" / "reservoir" / "instance_1.rddl"
    print(f"DOMAIN={DOMAIN}")
    print(f"INSTANCE={INSTANCE}")
    env = pyRDDLGym.make(domain=DOMAIN, instance=INSTANCE, vectorized=True)
    trainer = Train(
        model=env.model,
        action_space=env.action_space,
        lr=args.lr,
        hidden_sizes=tuple(args.hidden_sizes),
        seed=args.seed,
        simulator=TorchRDDLSimulator(env.model)
    )
    history = trainer.train_trajectory(
        iterations=args.iterations,
        print_every=args.print_every,
    )

    if history:
        final_metrics = history[-1]
        print(
            f"final return={final_metrics['return']:.4f} "
            f"after {int(final_metrics['iteration'])} updates"
        )


if __name__ == '__main__':
    main()

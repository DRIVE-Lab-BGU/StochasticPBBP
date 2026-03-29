  
from typing import Any, Dict, List, Optional, Optional, Sequence
import torch
import torch.nn as nn
from StochasticPBBP.core.Logic import FuzzyLogic
from StochasticPBBP.core.Rollout import TorchRollout


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
        return history

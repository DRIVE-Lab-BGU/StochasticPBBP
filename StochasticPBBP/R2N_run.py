from pathlib import Path

import pyRDDLGym
import torch
from StochasticPBBP.core.R2Trainer import R2Trainer
from StochasticPBBP.core.Rollout import TorchRollout
from StochasticPBBP.utils.Noise import AdditiveNoiseFactory
from StochasticPBBP.utils.Policies import NeuralStateFeedbackPolicy, StationaryMarkov
from StochasticPBBP.utils.R2Noise import R2GradientAdditiveNoise
from StochasticPBBP.core.Logic import (
    FuzzyLogic,
    ProductTNorm,
    SigmoidComparison,
    SoftRounding,
    SoftControlFlow,
    SoftRandomSampling,
)


def main() -> None:
    hidden_sizes = (128, 64)
    iterations = 200
    print_every = 10
    seed = 2
    package_root = Path(__file__).resolve().parent
    domain = package_root / "problems" / "reservoir" / "domain.rddl"
    instance = package_root / "problems" / "reservoir" / "instance_3.rddl"
    torch.manual_seed(seed)
    env = pyRDDLGym.make(
        domain=str(domain),
        instance=str(instance),
        vectorized=True,
    )
    horizon = int(env.model.horizon)
    horizon = 5
    template_rollout = TorchRollout(env.model, horizon=horizon)
    _, observation_template, _ = template_rollout.reset()
    ### TO UPDATE  ###
    # NERUAL STATE FEEDBACK POLICY
    #policy = StationaryMarkov(
    #    observation_template=observation_template,
    #    action_template=template_rollout.noop_actions,
    #    action_space=env.action_space,
    #    hidden_sizes=hidden_sizes)
    policy = NeuralStateFeedbackPolicy(
            observation_template=observation_template,
            action_template=template_rollout.noop_actions,
            hidden_sizes=hidden_sizes,
            action_space=env.action_space,seed=seed
        )


    analysis_noise = AdditiveNoiseFactory.create(
        noise_type='constant',
        std=0.0,
        source=template_rollout,
    )
    r2_noise = R2GradientAdditiveNoise(
        scale=1.0,
        eps=1e-6,
        min_std=1.0,
        max_std=5.0,
        alpha=0.1,
        norm_scope='global',
        step_score_aggregate='mean',
        normalization_quantile=0.95,
    )
    logic = FuzzyLogic(
            tnorm=ProductTNorm(),
            comparison=SigmoidComparison(weight=100.0),
            rounding=SoftRounding(weight=100.0),
            control=SoftControlFlow(weight=100.0),
            sampling=SoftRandomSampling(
                poisson_max_bins=100,
                binomial_max_bins=100,
                bernoulli_gumbel_softmax=False,),)

    trainer = R2Trainer(
        model=env.model,
        action_space=env.action_space,
        policy=policy,
        horizon=horizon,
        hidden_sizes=hidden_sizes,
        additive_noise=r2_noise,
        analysis_additive_noise=analysis_noise,
        logic=logic , seed=seed
    )

    history, trained_policy = trainer.train_trajectory(
        iterations=iterations,
        print_every=print_every,
    )

    final_metrics = history[-1]
    print('\nTraining finished.')
    print(f'iterations={len(history)}')
    # the update return is the return of the trajectory used for the update, 
    # which may be different from the return of the final policy after training (the analysis return).
    #  In practice, we find that the update return can be a noisy signal of training progress, while the analysis return is a more stable signal of final policy performance.
    print(f'last update return={final_metrics["update_return"]:.4f}') 
    print(f'last analysis return={final_metrics["analysis_return"]:.4f}')
    print(f'policy class={type(trained_policy).__name__}')
    print(f'r2 profile ready={r2_noise.has_profile}')
    print(f'gradient inf norm={r2_noise.last_gradient_inf_norm:.6f}')
    print(f'score quantile={r2_noise.last_score_quantile:.6f}')


if __name__ == '__main__':
    main()

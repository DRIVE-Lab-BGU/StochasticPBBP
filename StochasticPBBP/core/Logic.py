"""Torch port of pyRDDLGym_jax core logic utilities with soft relaxations."""

from abc import ABCMeta, abstractmethod
import random
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

########################################################
### here in the sigment we define abstract decorators i convert evrey input x to tensor
# to make sure that all operations are done in torch tensors
# maybe it is not necessary 
########################################################

## to do 
# 1  understand the argmax of sigmoid comparison
# understand  _get_generator in line 437
## _torch_wrapped_calc_poisson_gumbel_softmax in line 477
#  nee to go back SoftRandomSampling class and understand it better

# to ask: how godel tnorm is different from product tnorm in practice?











# this helper creates a tensor of indices along a given axis, broadcast to the desired shape
# e.g., shape=(2,3,4), axis=1 -> tensor([[[0,0,0,0],[1,1,1,1],[2,2,2,2]],
#                                     [[0,0,0,0],[1,1,1,1],[2,2,2,2]]]) 

def enumerate_literals(shape: Tuple[int, ...], axis: int, dtype: torch.dtype = torch.int32,
                      device=None) -> torch.Tensor:
    """Create a tensor of indices along the given axis, broadcast to `shape`."""
    axis = axis % len(shape)  # keep axis in range for negative values
    literals = torch.arange(shape[axis], dtype=dtype, device=device)
    view_shape = [1] * len(shape)
    view_shape[axis] = shape[axis]
    literals = literals.view(*view_shape)  # reshape for broadcasting
    return literals.expand(*shape)

# this helper reduces a tensor along multiple axes using the provided reduction function
# e.g., torch.min, torch.max, torch.sum, for torch.prod
def _reduce_dims(tensor: torch.Tensor, axes: Union[int, Sequence[int]],
                 reduce_fn: Callable[[torch.Tensor, int], torch.Tensor]) -> torch.Tensor:
    axes_tuple = tuple(axes) if isinstance(axes, (list, tuple)) else (axes,)
    result = tensor
    for ax in sorted(axes_tuple, reverse=True):  # reduce from last to keep axis validity
        result = reduce_fn(result, ax)
    return result


# ===========================================================================
# RELATIONAL OPERATIONS
# - abstract class
# - sigmoid comparison
# ===========================================================================
# here we define Comparison abstract base class,
#  its mean that we must implement its methods in subclasses
# for example if we want to use greater_equal, greater, equal, sgn, argmax methods
# we must implement them in the subclass

# this class defines the interface for comparison operations and make shure that 
# every comaring (in subclasses) will be consistent with this interface

class Comparison(metaclass=ABCMeta):
    """Base class for approximate comparison operations."""

    @abstractmethod
    def greater_equal(self, id, init_params):
        pass

    @abstractmethod
    def greater(self, id, init_params):
        pass

    @abstractmethod
    def equal(self, id, init_params):
        pass

    @abstractmethod
    def sgn(self, id, init_params):
        pass

    @abstractmethod
    def argmax(self, id, init_params):
        pass
#
### here we define SigmoidComparison class inheriting from Comparison
# https://arxiv.org/abs/2110.05651
class SigmoidComparison(Comparison):
    """Comparison operations approximated using sigmoid functions."""

    def __init__(self, weight: float = 10.0) -> None:
        self.weight = float(weight)  # scaling factor for soft boundaries

    def greater_equal(self, id, init_params):
        id_ = str(id)
        init_params[id_] = self.weight  # stash per-call weight so params stays mutable

        def _torch_wrapped_calc_greater_equal_approx(x, y, params):
            x_t = torch.as_tensor(x)
            y_t = torch.as_tensor(y, device=x_t.device, dtype=x_t.dtype)
            weight = torch.as_tensor(params[id_], dtype=x_t.dtype, device=x_t.device)
            # take the value of the sigmoid at (x - y)*weight
            gre_eq = torch.sigmoid(weight * (x_t - y_t))
            return gre_eq, params

        return _torch_wrapped_calc_greater_equal_approx

    def greater(self, id, init_params):
        return self.greater_equal(id, init_params)

    def equal(self, id, init_params):
        id_ = str(id)
        init_params[id_] = self.weight  # tighten/loosen equality sharpness

        def _torch_wrapped_calc_equal_approx(x, y, params):
            x_t = torch.as_tensor(x)
            y_t = torch.as_tensor(y, device=x_t.device, dtype=x_t.dtype)
            weight = torch.as_tensor(params[id_], dtype=x_t.dtype, device=x_t.device)
            equal = 1.0 - torch.tanh(weight * (y_t - x_t)).pow(2)
            return equal, params

        return _torch_wrapped_calc_equal_approx

    def sgn(self, id, init_params):
        id_ = str(id)
        init_params[id_] = self.weight  # same slope shared across calls

        def _torch_wrapped_calc_sgn_approx(x, params):
            x_t = torch.as_tensor(x)
            weight = torch.as_tensor(params[id_], dtype=x_t.dtype, device=x_t.device)
            sgn = torch.tanh(weight * x_t)
            return sgn, params

        return _torch_wrapped_calc_sgn_approx
######################################################################################################
## returen argmax function i dont get it##############################################################
#######################################################################################################
    # https://arxiv.org/abs/2110.05651
    def argmax(self, id, init_params):
        id_ = str(id)
        init_params[id_] = self.weight  # reuse weight to control softmax temperature

        def _torch_wrapped_calc_argmax_approx(x, axis, params):
            x_t = torch.as_tensor(x)
            axis = axis % x_t.dim()
            weight = torch.as_tensor(params[id_], dtype=x_t.dtype, device=x_t.device)
            literals = enumerate_literals(tuple(x_t.shape), axis=axis,
                                          dtype=x_t.dtype, device=x_t.device)
            softmax = torch.softmax(weight * x_t, dim=axis)  # temperature-controlled soft argmax
            sample = torch.sum(literals * softmax, dim=axis)
            return sample, params

        return _torch_wrapped_calc_argmax_approx
    # string representation of the class
    def __str__(self) -> str:
        return f'Sigmoid comparison with weight {self.weight}'


# ===========================================================================
# ROUNDING OPERATIONS
# - abstract class
# - soft rounding
# ===========================================================================

class Rounding(metaclass=ABCMeta):
    """Base class for approximate rounding operations."""

    @abstractmethod
    def floor(self, id, init_params):
        pass

    @abstractmethod
    def round(self, id, init_params):
        pass


class SoftRounding(Rounding):
    """Rounding operations approximated using soft operations."""

    def __init__(self, weight: float = 10.0) -> None:
        self.weight = float(weight)  # controls steepness of soft transitions

    # https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/bijectors/Softfloor

    
    def floor(self, id, init_params):
        """Soft floor that does not rely on mutable init_params keys."""
        def _torch_wrapped_calc_floor_approx(x, params):
            x_t = torch.as_tensor(x)
            param = torch.as_tensor(self.weight, dtype=x_t.dtype, device=x_t.device)
            denom = torch.tanh(param / 4.0)
            floor = (torch.sigmoid(param * (x_t - torch.floor(x_t) - 1.0)) -
                     torch.sigmoid(-param / 2.0)) / denom + torch.floor(x_t)
            return floor, params

        return _torch_wrapped_calc_floor_approx

    def round(self, id, init_params):
        """Soft round that does not rely on mutable init_params keys."""
        def _torch_wrapped_calc_round_approx(x, params):
            x_t = torch.as_tensor(x)
            param = torch.as_tensor(self.weight, dtype=x_t.dtype, device=x_t.device)
            m = torch.floor(x_t) + 0.5
            rounded = m + 0.5 * torch.tanh(param * (x_t - m)) / torch.tanh(param / 2.0)
            return rounded, params

        return _torch_wrapped_calc_round_approx

    def __str__(self) -> str:
        return f'SoftFloor and SoftRound with weight {self.weight}'




# ===========================================================================
# LOGICAL COMPLEMENT
# - abstract class
# - standard complement
# ===========================================================================

class Complement(metaclass=ABCMeta):
    """Base class for approximate logical complement operations."""

    @abstractmethod
    def __call__(self, id, init_params):
        pass


class StandardComplement(Complement):
    """The standard approximate logical complement given by x -> 1 - x."""

    @staticmethod
    def _torch_wrapped_calc_not_approx(x, params):
        return 1.0 - x, params

    def __call__(self, id, init_params):
        return self._torch_wrapped_calc_not_approx

    def __str__(self) -> str:
        return 'Standard complement'


# ===========================================================================
# TNORMS
# - abstract tnorm
# - product tnorm
# - Godel tnorm
# - Lukasiewicz tnorm
# - Yager(p) tnorm
# ===========================================================================







class TNorm(metaclass=ABCMeta):
    """Base class for fuzzy differentiable t-norms."""

    @abstractmethod
    def norm(self, id, init_params):
        """Elementwise t-norm of x and y."""
        pass

    @abstractmethod
    def norms(self, id, init_params):
        """T-norm computed for tensor x along axis."""
        pass


class ProductTNorm(TNorm):
    """Product t-norm given by the expression (x, y) -> x * y."""

    @staticmethod
    def _torch_wrapped_calc_and_approx(x, y, params):
        return x * y, params

    def norm(self, id, init_params):
        return self._torch_wrapped_calc_and_approx

    @staticmethod
    def _torch_wrapped_calc_forall_approx(x, axis, params):
        # fold product across possibly multiple axes
        axes = tuple(axis) if isinstance(axis, (list, tuple)) else (axis,)
        # adding an option for axis to be a list of axes - different from original
        result = x
        for ax in sorted(axes, reverse=True):
            result = torch.prod(result, dim=ax)
        return result, params

    def norms(self, id, init_params):
        return self._torch_wrapped_calc_forall_approx

    def __str__(self) -> str:
        return 'Product t-norm'



class GodelTNorm(TNorm):
    """Godel t-norm given by the expression (x, y) -> min(x, y)."""

    @staticmethod
    def _torch_wrapped_calc_and_approx(x, y, params):
        return torch.minimum(x, y), params

    def norm(self, id, init_params):
        return self._torch_wrapped_calc_and_approx

    @staticmethod
    def _torch_wrapped_calc_forall_approx(x, axis, params):
        # apply min along each axis sequentially
        axes = tuple(axis) if isinstance(axis, (list, tuple)) else (axis,)
       # adding an option for axis to be a list of axes - different from original
        result = x
        for ax in sorted(axes, reverse=True):
            result = torch.min(result, dim=ax).values
        return result, params

    def norms(self, id, init_params):
        return self._torch_wrapped_calc_forall_approx

    def __str__(self) -> str:
        return 'Godel t-norm'


class LukasiewiczTNorm(TNorm):
    """Lukasiewicz t-norm given by the expression (x, y) -> max(x + y - 1, 0)."""

    @staticmethod
    def _torch_wrapped_calc_and_approx(x, y, params):
        land = F.relu(x + y - 1.0)
        return land, params

    def norm(self, id, init_params):
        return self._torch_wrapped_calc_and_approx

    @staticmethod
    def _torch_wrapped_calc_forall_approx(x, axis, params):
        forall = F.relu(torch.sum(x - 1.0, dim=axis) + 1.0)
        return forall, params

    def norms(self, id, init_params):
        return self._torch_wrapped_calc_forall_approx

    def __str__(self) -> str:
        return 'Lukasiewicz t-norm'


class YagerTNorm(TNorm):
    """Yager t-norm given by the expression
    (x, y) -> max(1 - ((1 - x)^p + (1 - y)^p)^(1/p))."""

    def __init__(self, p: float = 2.0) -> None:
        self.p = float(p)

    def norm(self, id, init_params):
        id_ = str(id)
        init_params[id_] = self.p

        def _torch_wrapped_calc_and_approx(x, y, params):
            base = F.relu(1.0 - torch.stack([x, y], dim=0))
            arg = torch.linalg.norm(base, ord=params[id_], dim=0)
            land = F.relu(1.0 - arg)
            return land, params

        return _torch_wrapped_calc_and_approx

    def norms(self, id, init_params):
        id_ = str(id)
        init_params[id_] = self.p

        def _torch_wrapped_calc_forall_approx(x, axis, params):
            axes: Sequence[int]
            axes = tuple(axis) if isinstance(axis, (list, tuple)) else (axis,)
            arg = F.relu(1.0 - x)
            for ax in sorted(axes, reverse=True):
                arg = torch.linalg.norm(arg, ord=params[id_], dim=ax)
            forall = F.relu(1.0 - arg)
            return forall, params

        return _torch_wrapped_calc_forall_approx

    def __str__(self) -> str:
        return f'Yager({self.p}) t-norm'

########################## here#####################################




#  here i add something new

# ===========================================================================
# RANDOM SAMPLING
# - abstract sampler
# - Gumbel-softmax sampler
# - determinization
# ===========================================================================

class RandomSampling(metaclass=ABCMeta):
    """Describes how non-reparameterizable random variables are sampled."""

    @abstractmethod
    def discrete(self, id, init_params, logic):
        pass

    @abstractmethod
    def poisson(self, id, init_params, logic):
        pass

    @abstractmethod
    def binomial(self, id, init_params, logic):
        pass

    @abstractmethod
    def negative_binomial(self, id, init_params, logic):
        pass

    @abstractmethod
    def geometric(self, id, init_params, logic):
        pass

    @abstractmethod
    def bernoulli(self, id, init_params, logic):
        pass

    def __str__(self) -> str:
        return 'RandomSampling'


class SoftRandomSampling(RandomSampling):
    """Random sampling of discrete variables using Gumbel-softmax trick."""

    def __init__(self, poisson_max_bins: int = 100,
                 poisson_min_cdf: float = 0.999,
                 poisson_exp_sampling: bool = True,
                 binomial_max_bins: int = 100,
                 bernoulli_gumbel_softmax: bool = False) -> None:
        self.poisson_bins = poisson_max_bins
        self.poisson_min_cdf = poisson_min_cdf
        self.poisson_exp_method = poisson_exp_sampling
        self.binomial_bins = binomial_max_bins
        self.bernoulli_gumbel_softmax = bernoulli_gumbel_softmax

    @staticmethod
    def _get_generator(key: Any) -> Union[torch.Generator, None]:
        return key if isinstance(key, torch.Generator) else None  # allow caller to pass RNG

    # https://arxiv.org/pdf/1611.01144
    def discrete(self, id, init_params, logic):
        argmax_approx = logic.argmax(id, init_params)

        def _torch_wrapped_calc_discrete_gumbel_softmax(key, prob, params):
            gen = self._get_generator(key)
            prob_t = torch.as_tensor(prob, dtype=logic.REAL)
            U = torch.rand(prob_t.shape, generator=gen, device=prob_t.device, dtype=prob_t.dtype)
            gumbel = -torch.log(-torch.log(U.clamp_min(logic.eps)))  # standard Gumbel(0,1)
            sample = gumbel + torch.log(prob_t + logic.eps)
            return argmax_approx(sample, axis=-1, params=params)

        return _torch_wrapped_calc_discrete_gumbel_softmax

    def _poisson_gumbel_softmax(self, id, init_params, logic):
        argmax_approx = logic.argmax(id, init_params)
        ##### here i need to understnad the poisson gumbel softmax
        def _torch_wrapped_calc_poisson_gumbel_softmax(key, rate, params):
            # gen is torch.Generator object for random number generation
            gen = self._get_generator(key)
            rate_t = torch.as_tensor(rate, dtype=logic.REAL)
            ks = torch.arange(self.poisson_bins, dtype=logic.REAL, device=rate_t.device)
            view_shape = [1] * rate_t.dim() + [self.poisson_bins]
            ks = ks.view(*view_shape)
            rate_expanded = rate_t.unsqueeze(-1)
            log_prob = ks * torch.log(rate_expanded + logic.eps) - rate_expanded - torch.lgamma(ks + 1)
            U = torch.rand_like(log_prob, generator=gen)
            gumbel = -torch.log(-torch.log(U.clamp_min(logic.eps)))  # Gumbel noise
            sample = gumbel + log_prob
            return argmax_approx(sample, axis=-1, params=params)

        return _torch_wrapped_calc_poisson_gumbel_softmax

    def _poisson_exponential(self, id, init_params, logic):
        less_approx = logic.less(id, init_params)

        def _torch_wrapped_calc_poisson_exponential(key, rate, params):
            gen = self._get_generator(key)

            rate_t = torch.as_tensor(rate, dtype=logic.REAL)
            shape = (self.poisson_bins,) + tuple(rate_t.shape)
            U = torch.rand(shape, generator=gen, device=rate_t.device, dtype=logic.REAL)
            Exp1 = -torch.log(U.clamp_min(logic.eps))  # exponential(1) samples
            delta_t = Exp1 / rate_t.unsqueeze(0)
            times = torch.cumsum(delta_t, dim=0)
            indicator, params = less_approx(times, 1.0, params)
            sample = torch.sum(indicator, dim=0)
            return sample, params

        return _torch_wrapped_calc_poisson_exponential
    # normal approximation to Poisson: Poisson(rate) -> Normal(rate, rate)
    def _poisson_normal_approx(self, logic):
        def _torch_wrapped_calc_poisson_normal_approx(key, rate, params):
            gen = self._get_generator(key)
            rate_t = torch.as_tensor(rate, dtype=logic.REAL)
            normal = torch.randn(rate_t.shape, generator=gen, device=rate_t.device, dtype=logic.REAL)
            sample = rate_t + torch.sqrt(rate_t) * normal  # Normal(rate, rate)
            return sample, params

        return _torch_wrapped_calc_poisson_normal_approx

    def poisson(self, id, init_params, logic):
        if self.poisson_exp_method:
            _torch_wrapped_calc_poisson_diff = self._poisson_exponential(id, init_params, logic)
        else:
            _torch_wrapped_calc_poisson_diff = self._poisson_gumbel_softmax(id, init_params, logic)
        _torch_wrapped_calc_poisson_normal = self._poisson_normal_approx(logic)

        def _poisson_cdf_bins(rate_tensor: torch.Tensor) -> torch.Tensor:
            try:
                dist = torch.distributions.Poisson(rate_tensor)
                return dist.cdf(torch.tensor(self.poisson_bins, dtype=logic.REAL, device=rate_tensor.device))
            except Exception:
                pass
            if hasattr(torch.special, "gammainc"):
                k_plus_one = torch.as_tensor(self.poisson_bins + 1, dtype=logic.REAL, device=rate_tensor.device)
                lower = torch.special.gammainc(k_plus_one, rate_tensor)
                gamma_func = torch.exp(torch.lgamma(k_plus_one))
                lower_reg = lower / gamma_func
                return 1.0 - lower_reg
            return torch.ones_like(rate_tensor)  # fallback avoids crashing when cdf unavailable

        def _torch_wrapped_calc_poisson_approx(key, rate, params):
            rate_t = torch.as_tensor(rate, dtype=logic.REAL)
            if self.poisson_bins > 0:
                cuml_prob = _poisson_cdf_bins(rate_t)
                small_rate = cuml_prob >= self.poisson_min_cdf  # truncate if mass within bins
                small_sample, params = _torch_wrapped_calc_poisson_diff(key, rate_t, params)
                large_sample, params = _torch_wrapped_calc_poisson_normal(key, rate_t, params)
                sample = torch.where(small_rate, small_sample, large_sample)
                return sample, params
            else:
                return _torch_wrapped_calc_poisson_normal(key, rate_t, params)

        return _torch_wrapped_calc_poisson_approx

    def _binomial_normal_approx(self, logic):
        def _torch_wrapped_calc_binomial_normal_approx(key, trials, prob, params):
            gen = self._get_generator(key)
            trials_t = torch.as_tensor(trials, dtype=logic.REAL)
            prob_t = torch.as_tensor(prob, dtype=logic.REAL, device=trials_t.device)
            normal = torch.randn(trials_t.shape, generator=gen, device=trials_t.device, dtype=logic.REAL)
            mean = trials_t * prob_t
            std = torch.sqrt(trials_t * prob_t * (1.0 - prob_t))
            sample = mean + std * normal
            return sample, params

        return _torch_wrapped_calc_binomial_normal_approx

    def _binomial_gumbel_softmax(self, id, init_params, logic):
        argmax_approx = logic.argmax(id, init_params)

        def _torch_wrapped_calc_binomial_gumbel_softmax(key, trials, prob, params):
            gen = self._get_generator(key)
            trials_t = torch.as_tensor(trials, dtype=logic.REAL)
            prob_t = torch.as_tensor(prob, dtype=logic.REAL, device=trials_t.device)
            ks = torch.arange(self.binomial_bins, dtype=logic.REAL, device=trials_t.device)
            view_shape = [1] * trials_t.dim() + [self.binomial_bins]
            ks = ks.view(*view_shape)
            trials_exp = trials_t.unsqueeze(-1)
            prob_exp = prob_t.unsqueeze(-1)
            in_support = ks <= trials_exp
            ks = torch.minimum(ks, trials_exp)
            log_prob = (torch.lgamma(trials_exp + 1) -
                        torch.lgamma(ks + 1) -
                        torch.lgamma(trials_exp - ks + 1) +
                        ks * torch.log(prob_exp + logic.eps) +
                        (trials_exp - ks) * torch.log1p(-prob_exp + logic.eps))
            log_prob = torch.where(in_support, log_prob,
                                   torch.log(torch.tensor(logic.eps, dtype=log_prob.dtype,
                                                          device=log_prob.device)))
            U = torch.rand_like(log_prob, generator=gen)
            gumbel = -torch.log(-torch.log(U.clamp_min(logic.eps)))  # Gumbel noise
            sample = gumbel + log_prob
            return argmax_approx(sample, axis=-1, params=params)

        return _torch_wrapped_calc_binomial_gumbel_softmax

    def binomial(self, id, init_params, logic):
        _torch_wrapped_calc_binomial_normal = self._binomial_normal_approx(logic)
        _torch_wrapped_calc_binomial_gs = self._binomial_gumbel_softmax(id, init_params, logic)

        def _torch_wrapped_calc_binomial_approx(key, trials, prob, params):
            trials_t = torch.as_tensor(trials, dtype=logic.REAL)
            prob_t = torch.as_tensor(prob, dtype=logic.REAL, device=trials_t.device)
            small_trials = trials_t < self.binomial_bins  # switch to normal approx when large
            small_sample, params = _torch_wrapped_calc_binomial_gs(key, trials_t, prob_t, params)
            large_sample, params = _torch_wrapped_calc_binomial_normal(key, trials_t, prob_t, params)
            sample = torch.where(small_trials, small_sample, large_sample)
            return sample, params

        return _torch_wrapped_calc_binomial_approx

    def negative_binomial(self, id, init_params, logic):
        poisson_approx = self.poisson(id, init_params, logic)

        def _torch_wrapped_calc_negative_binomial_approx(key, trials, prob, params):
            gen = self._get_generator(key)
            trials_t = torch.as_tensor(trials, dtype=logic.REAL)
            prob_t = torch.as_tensor(prob, dtype=logic.REAL, device=trials_t.device)
            # gamma-poisson mixture for Negative Binomial
            Gamma = torch.distributions.Gamma(trials_t, torch.tensor(1.0, dtype=logic.REAL,
                                                                     device=trials_t.device)).sample(generator=gen)
            scale = (1.0 - prob_t) / prob_t
            poisson_rate = scale * Gamma
            return poisson_approx(key, poisson_rate, params)

        return _torch_wrapped_calc_negative_binomial_approx

    def geometric(self, id, init_params, logic):
        approx_floor = logic.floor(id, init_params)

        def _torch_wrapped_calc_geometric_approx(key, prob, params):
            gen = self._get_generator(key)
            prob_t = torch.as_tensor(prob, dtype=logic.REAL)
            U = torch.rand(prob_t.shape, generator=gen, device=prob_t.device, dtype=logic.REAL)
            # inverse-CDF using soft floor
            floor, params = approx_floor(torch.log1p(-U) / torch.log1p(-prob_t + logic.eps), params)
            sample = floor + 1
            return sample, params

        return _torch_wrapped_calc_geometric_approx

    def _bernoulli_uniform(self, id, init_params, logic):
        less_approx = logic.less(id, init_params)

        def _torch_wrapped_calc_bernoulli_uniform(key, prob, params):
            gen = self._get_generator(key)
            prob_t = torch.as_tensor(prob, dtype=logic.REAL)
            U = torch.rand(prob_t.shape, generator=gen, device=prob_t.device, dtype=logic.REAL)
            return less_approx(U, prob_t, params)

        return _torch_wrapped_calc_bernoulli_uniform

    def _bernoulli_gumbel_softmax(self, id, init_params, logic):
        discrete_approx = self.discrete(id, init_params, logic)

        def _torch_wrapped_calc_bernoulli_gumbel_softmax(key, prob, params):
            prob_t = torch.as_tensor(prob, dtype=logic.REAL)
            prob = torch.stack([1.0 - prob_t, prob_t], dim=-1)  # two-class categorical
            return discrete_approx(key, prob, params)

        return _torch_wrapped_calc_bernoulli_gumbel_softmax

    def bernoulli(self, id, init_params, logic):
        if self.bernoulli_gumbel_softmax:
            return self._bernoulli_gumbel_softmax(id, init_params, logic)
        else:
            return self._bernoulli_uniform(id, init_params, logic)

    def __str__(self) -> str:
        return 'SoftRandomSampling'


class Determinization(RandomSampling):
    """Random sampling of variables using their deterministic mean estimate."""

    @staticmethod
    def _torch_wrapped_calc_discrete_determinized(key, prob, params):
        prob_t = torch.as_tensor(prob)
        literals = enumerate_literals(tuple(prob_t.shape), axis=-1, dtype=prob_t.dtype, device=prob_t.device)
        sample = torch.sum(literals * prob_t, dim=-1)  # expected value of categorical
        return sample, params

    def discrete(self, id, init_params, logic):
        return self._torch_wrapped_calc_discrete_determinized

    @staticmethod
    def _torch_wrapped_calc_poisson_determinized(key, rate, params):
        return rate, params

    def poisson(self, id, init_params, logic):
        return self._torch_wrapped_calc_poisson_determinized

    @staticmethod
    def _torch_wrapped_calc_binomial_determinized(key, trials, prob, params):
        sample = trials * prob
        return sample, params

    def binomial(self, id, init_params, logic):
        return self._torch_wrapped_calc_binomial_determinized

    @staticmethod
    def _torch_wrapped_calc_negative_binomial_determinized(key, trials, prob, params):
        sample = trials * ((1.0 / prob) - 1.0)
        return sample, params

    def negative_binomial(self, id, init_params, logic):
        return self._torch_wrapped_calc_negative_binomial_determinized

    @staticmethod
    def _torch_wrapped_calc_geometric_determinized(key, prob, params):
        sample = 1.0 / prob
        return sample, params

    def geometric(self, id, init_params, logic):
        return self._torch_wrapped_calc_geometric_determinized

    @staticmethod
    def _torch_wrapped_calc_bernoulli_determinized(key, prob, params):
        sample = prob
        return sample, params

    def bernoulli(self, id, init_params, logic):
        return self._torch_wrapped_calc_bernoulli_determinized

    def __str__(self) -> str:
        return 'Deterministic'


# ===========================================================================
# CONTROL FLOW
# - soft flow
# ===========================================================================

class ControlFlow(metaclass=ABCMeta):
    """A base class for control flow, including if and switch statements."""

    @abstractmethod
    def if_then_else(self, id, init_params):
        pass

    @abstractmethod
    def switch(self, id, init_params):
        pass


class SoftControlFlow(ControlFlow):
    """Soft control flow using a probabilistic interpretation."""

    def __init__(self, weight: float = 10.0) -> None:
        self.weight = float(weight)  # temperature for soft switch

    @staticmethod
    def _torch_wrapped_calc_if_then_else_soft(c, a, b, params):
        sample = c * a + (1.0 - c) * b  # convex combination by soft predicate
        return sample, params

    def if_then_else(self, id, init_params):
        return self._torch_wrapped_calc_if_then_else_soft

    def switch(self, id, init_params):
        id_ = str(id)
        init_params[id_] = self.weight  # store temperature per call

        def _torch_wrapped_calc_switch_soft(pred, cases, params):
            cases_t = torch.as_tensor(cases)
            pred_t = torch.as_tensor(pred, dtype=cases_t.dtype, device=cases_t.device)
            literals = enumerate_literals(tuple(cases_t.shape), axis=0,
                                          dtype=cases_t.dtype, device=cases_t.device)
            pred_t = pred_t.unsqueeze(0).expand_as(cases_t)
            proximity = -(pred_t - literals).pow(2)  # favor close indices
            softcase = torch.softmax(params[id_] * proximity, dim=0)  # temperature from stored weight
            sample = torch.sum(cases_t * softcase, dim=0)
            return sample, params

        return _torch_wrapped_calc_switch_soft

    def __str__(self) -> str:
        return f'Soft control flow with weight {self.weight}'


# ===========================================================================
# LOGIC
# - exact logic
# - fuzzy logic
# ===========================================================================

class Logic(metaclass=ABCMeta):
    """A base class for representing logic computations in torch."""

    def __init__(self, use64bit: bool = False) -> None:
        self.set_use64bit(use64bit)

    def summarize_hyperparameters(self) -> str:
        return (f'model relaxation:\n'
                f'    use_64_bit    ={self.use64bit}')

    def set_use64bit(self, use64bit: bool) -> None:
        """Toggles whether or not torch will use 64 bit precision."""
        self.use64bit = use64bit
        if use64bit:
            self.REAL = torch.float64
            self.INT = torch.int64
        else:
            self.REAL = torch.float32
            self.INT = torch.int32

    @staticmethod
    def wrap_logic(func):
        # wrap a pure function into the (id, init_params) signature expected by callers
        def exact_func(id, init_params):
            return func
        return exact_func

    def get_operator_dicts(self) -> Dict[str, Union[Callable, Dict[str, Callable]]]:
        """Returns a dictionary of all operators in the current logic."""
        return {
            'negative': self.wrap_logic(ExactLogic.exact_unary_function(torch.neg)),
            'arithmetic': {
                '+': self.wrap_logic(ExactLogic.exact_binary_function(torch.add)),
                '-': self.wrap_logic(ExactLogic.exact_binary_function(torch.sub)),
                '*': self.wrap_logic(ExactLogic.exact_binary_function(torch.mul)),
                '/': self.wrap_logic(ExactLogic.exact_binary_function(torch.div))
            },
            'relational': {
                '>=': self.greater_equal,
                '<=': self.less_equal,
                '<': self.less,
                '>': self.greater,
                '==': self.equal,
                '~=': self.not_equal
            },
            'logical_not': self.logical_not,
            'logical': {
                '^': self.logical_and,
                '&': self.logical_and,
                '|': self.logical_or,
                '~': self.xor,
                '=>': self.implies,
                '<=>': self.equiv
            },
            'aggregation': {
                'sum': self.wrap_logic(ExactLogic.exact_aggregation(torch.sum)),
                'avg': self.wrap_logic(ExactLogic.exact_aggregation(torch.mean)),
                'prod': self.wrap_logic(ExactLogic.exact_aggregation(torch.prod)),
                'minimum': self.wrap_logic(
                    ExactLogic.exact_aggregation(
                        lambda x, dim: _reduce_dims(x, dim, lambda t, ax: torch.min(t, dim=ax).values))),
                'maximum': self.wrap_logic(
                    ExactLogic.exact_aggregation(
                        lambda x, dim: _reduce_dims(x, dim, lambda t, ax: torch.max(t, dim=ax).values))),
                'forall': self.forall,
                'exists': self.exists,
                'argmin': self.argmin,
                'argmax': self.argmax
            },
            'unary': {
                'abs': self.wrap_logic(ExactLogic.exact_unary_function(torch.abs)),
                'sgn': self.sgn,
                'round': self.round,
                'floor': self.floor,
                'ceil': self.ceil,
                'cos': self.wrap_logic(ExactLogic.exact_unary_function(torch.cos)),
                'sin': self.wrap_logic(ExactLogic.exact_unary_function(torch.sin)),
                'tan': self.wrap_logic(ExactLogic.exact_unary_function(torch.tan)),
                'acos': self.wrap_logic(ExactLogic.exact_unary_function(torch.acos)),
                'asin': self.wrap_logic(ExactLogic.exact_unary_function(torch.asin)),
                'atan': self.wrap_logic(ExactLogic.exact_unary_function(torch.atan)),
                'cosh': self.wrap_logic(ExactLogic.exact_unary_function(torch.cosh)),
                'sinh': self.wrap_logic(ExactLogic.exact_unary_function(torch.sinh)),
                'tanh': self.wrap_logic(ExactLogic.exact_unary_function(torch.tanh)),
                'exp': self.wrap_logic(ExactLogic.exact_unary_function(torch.exp)),
                'ln': self.wrap_logic(ExactLogic.exact_unary_function(torch.log)),
                'sqrt': self.sqrt,
                'lngamma': self.wrap_logic(ExactLogic.exact_unary_function(torch.lgamma)),
                'gamma': self.wrap_logic(
                    ExactLogic.exact_unary_function(
                        torch.special.gamma if hasattr(torch.special, 'gamma')
                        else lambda x: torch.exp(torch.lgamma(x)))),
            },
            'binary': {
                'div': self.div,
                'mod': self.mod,
                'fmod': self.mod,
                'min': self.wrap_logic(ExactLogic.exact_binary_function(torch.minimum)),
                'max': self.wrap_logic(ExactLogic.exact_binary_function(torch.maximum)),
                'pow': self.wrap_logic(ExactLogic.exact_binary_function(torch.pow)),
                'log': self.wrap_logic(ExactLogic.exact_binary_log),
                'hypot': self.wrap_logic(
                    ExactLogic.exact_binary_function(
                        torch.hypot if hasattr(torch, 'hypot')
                        else lambda a, b: torch.sqrt(a * a + b * b))),
            },
            'control': {
                'if': self.control_if,
                'switch': self.control_switch
            },
            'sampling': {
                'Bernoulli': self.bernoulli,
                'Discrete': self.discrete,
                'Poisson': self.poisson,
                'Geometric': self.geometric,
                'Binomial': self.binomial,
                'NegativeBinomial': self.negative_binomial
            }
        }

    # ===========================================================================
    # logical operators
    # ===========================================================================

    @abstractmethod
    def logical_and(self, id, init_params):
        pass

    @abstractmethod
    def logical_not(self, id, init_params):
        pass

    @abstractmethod
    def logical_or(self, id, init_params):
        pass

    @abstractmethod
    def xor(self, id, init_params):
        pass

    @abstractmethod
    def implies(self, id, init_params):
        pass

    @abstractmethod
    def equiv(self, id, init_params):
        pass

    @abstractmethod
    def forall(self, id, init_params):
        pass

    @abstractmethod
    def exists(self, id, init_params):
        pass

    # ===========================================================================
    # comparison operators
    # ===========================================================================

    @abstractmethod
    def greater_equal(self, id, init_params):
        pass

    @abstractmethod
    def greater(self, id, init_params):
        pass

    @abstractmethod
    def less_equal(self, id, init_params):
        pass

    @abstractmethod
    def less(self, id, init_params):
        pass

    @abstractmethod
    def equal(self, id, init_params):
        pass

    @abstractmethod
    def not_equal(self, id, init_params):
        pass

    # ===========================================================================
    # special functions
    # ===========================================================================

    @abstractmethod
    def sgn(self, id, init_params):
        pass

    @abstractmethod
    def floor(self, id, init_params):
        pass

    @abstractmethod
    def round(self, id, init_params):
        pass

    @abstractmethod
    def ceil(self, id, init_params):
        pass

    @abstractmethod
    def div(self, id, init_params):
        pass

    @abstractmethod
    def mod(self, id, init_params):
        pass

    @abstractmethod
    def sqrt(self, id, init_params):
        pass

    # ===========================================================================
    # indexing
    # ===========================================================================

    @abstractmethod
    def argmax(self, id, init_params):
        pass

    @abstractmethod
    def argmin(self, id, init_params):
        pass

    # ===========================================================================
    # control flow
    # ===========================================================================

    @abstractmethod
    def control_if(self, id, init_params):
        pass

    @abstractmethod
    def control_switch(self, id, init_params):
        pass

    # ===========================================================================
    # random variables
    # ===========================================================================

    @abstractmethod
    def discrete(self, id, init_params):
        pass

    @abstractmethod
    def bernoulli(self, id, init_params):
        pass

    @abstractmethod
    def poisson(self, id, init_params):
        pass

    @abstractmethod
    def geometric(self, id, init_params):
        pass

    @abstractmethod
    def binomial(self, id, init_params):
        pass

    @abstractmethod
    def negative_binomial(self, id, init_params):
        pass


class ExactLogic(Logic):
    """A class representing exact logic in torch."""

    @staticmethod
    def exact_unary_function(op):
        def _torch_wrapped_calc_unary_function_exact(x, params):
            return op(x), params
        return _torch_wrapped_calc_unary_function_exact

    @staticmethod
    def exact_binary_function(op):
        def _torch_wrapped_calc_binary_function_exact(x, y, params):
            return op(x, y), params
        return _torch_wrapped_calc_binary_function_exact

    @staticmethod
    def exact_aggregation(op):
        def _torch_wrapped_calc_aggregation_exact(x, axis, params):
            axis_tuple = tuple(axis) if isinstance(axis, (list, tuple)) else axis  # support multi-axis reductions
            return op(x, dim=axis_tuple), params
        return _torch_wrapped_calc_aggregation_exact

    # ===========================================================================
    # logical operators
    # ===========================================================================

    def logical_and(self, id, init_params):
        return self.exact_binary_function(torch.logical_and)

    def logical_not(self, id, init_params):
        return self.exact_unary_function(torch.logical_not)

    def logical_or(self, id, init_params):
        return self.exact_binary_function(torch.logical_or)

    def xor(self, id, init_params):
        return self.exact_binary_function(torch.logical_xor)

    @staticmethod
    def _torch_wrapped_calc_implies_exact(x, y, params):
        return torch.logical_or(torch.logical_not(x), y), params

    def implies(self, id, init_params):
        return self._torch_wrapped_calc_implies_exact

    def equiv(self, id, init_params):
        return self.exact_binary_function(torch.eq)

    def forall(self, id, init_params):
        return self.exact_aggregation(torch.all)

    def exists(self, id, init_params):
        return self.exact_aggregation(torch.any)

    # ===========================================================================
    # comparison operators
    # ===========================================================================

    def greater_equal(self, id, init_params):
        return self.exact_binary_function(torch.ge)

    def greater(self, id, init_params):
        return self.exact_binary_function(torch.gt)

    def less_equal(self, id, init_params):
        return self.exact_binary_function(torch.le)

    def less(self, id, init_params):
        return self.exact_binary_function(torch.lt)

    def equal(self, id, init_params):
        return self.exact_binary_function(torch.eq)

    def not_equal(self, id, init_params):
        return self.exact_binary_function(torch.ne)

    # ===========================================================================
    # special functions
    # ===========================================================================

    @staticmethod
    def exact_binary_log(x, y, params):
        return torch.log(x) / torch.log(y), params

    def sgn(self, id, init_params):
        return self.exact_unary_function(torch.sign)

    def floor(self, id, init_params):
        return self.exact_unary_function(torch.floor)

    def round(self, id, init_params):
        return self.exact_unary_function(torch.round)

    def ceil(self, id, init_params):
        return self.exact_unary_function(torch.ceil)

    def div(self, id, init_params):
        return self.exact_binary_function(torch.floor_divide)

    def mod(self, id, init_params):
        return self.exact_binary_function(torch.remainder)

    def sqrt(self, id, init_params):
        return self.exact_unary_function(torch.sqrt)

    # ===========================================================================
    # indexing
    # ===========================================================================

    def argmax(self, id, init_params):
        return self.exact_aggregation(torch.argmax)

    def argmin(self, id, init_params):
        return self.exact_aggregation(torch.argmin)

    # ===========================================================================
    # control flow
    # ===========================================================================

    @staticmethod
    def _torch_wrapped_calc_if_then_else_exact(c, a, b, params):
        return torch.where(c > 0.5, a, b), params

    def control_if(self, id, init_params):
        return self._torch_wrapped_calc_if_then_else_exact

    def control_switch(self, id, init_params):
        def _torch_wrapped_calc_switch_exact(pred, cases, params):
            cases_t = torch.as_tensor(cases)
            # `take_along_dim` requires int64/long indices even when the model
            # itself is running in 32-bit integer mode.
            pred_t = torch.as_tensor(pred, dtype=torch.long, device=cases_t.device)
            pred_t = pred_t.unsqueeze(0)
            sample = torch.take_along_dim(cases_t, pred_t, dim=0)  # deterministic gather
            assert sample.shape[0] == 1
            return sample[0, ...], params
        return _torch_wrapped_calc_switch_exact

    # ===========================================================================
    # random variables
    # ===========================================================================

    @staticmethod
    def _torch_wrapped_calc_discrete_exact(key, prob, params):
        prob_t = torch.as_tensor(prob)
        gen = key if isinstance(key, torch.Generator) else None
        sample = torch.multinomial(prob_t, num_samples=1, generator=gen).squeeze(-1)  # categorical draw
        sample = sample.to(prob_t.device, dtype=torch.int64)
        return sample, params

    def discrete(self, id, init_params):
        return self._torch_wrapped_calc_discrete_exact

    @staticmethod
    def _torch_wrapped_calc_bernoulli_exact(key, prob, params):
        gen = key if isinstance(key, torch.Generator) else None
        prob_t = torch.as_tensor(prob)
        return torch.bernoulli(prob_t, generator=gen), params

    def bernoulli(self, id, init_params):
        return self._torch_wrapped_calc_bernoulli_exact

    def poisson(self, id, init_params):
        def _torch_wrapped_calc_poisson_exact(key, rate, params):
            gen = key if isinstance(key, torch.Generator) else None
            rate_t = torch.as_tensor(rate, dtype=self.REAL)
            sample = torch.poisson(rate_t, generator=gen).to(self.INT)  # torch poisson supports generator
            return sample, params
        return _torch_wrapped_calc_poisson_exact

    def geometric(self, id, init_params):
        def _torch_wrapped_calc_geometric_exact(key, prob, params):
            gen = key if isinstance(key, torch.Generator) else None
            prob_t = torch.as_tensor(prob, dtype=self.REAL)
            U = torch.rand(prob_t.shape, generator=gen, device=prob_t.device, dtype=self.REAL)
            sample = torch.floor(torch.log1p(-U) / torch.log1p(-prob_t)) + 1  # inverse-CDF
            sample = sample.to(self.INT)
            return sample, params
        return _torch_wrapped_calc_geometric_exact

    def binomial(self, id, init_params):
        def _torch_wrapped_calc_binomial_exact(key, trials, prob, params):
            trials_t = torch.as_tensor(trials, dtype=self.REAL)
            prob_t = torch.as_tensor(prob, dtype=self.REAL, device=trials_t.device)
            dist = torch.distributions.Binomial(total_count=trials_t, probs=prob_t)  # vectorized Bin(n,p)
            sample = dist.sample().to(self.INT)
            return sample, params
        return _torch_wrapped_calc_binomial_exact

    def negative_binomial(self, id, init_params):
        def _torch_wrapped_calc_negative_binomial_exact(key, trials, prob, params):
            trials_t = torch.as_tensor(trials, dtype=self.REAL)
            prob_t = torch.as_tensor(prob, dtype=self.REAL, device=trials_t.device)
            dist = torch.distributions.NegativeBinomial(total_count=trials_t, probs=1.0 - prob_t)
            sample = dist.sample().to(self.INT)
            return sample, params
        return _torch_wrapped_calc_negative_binomial_exact


class FuzzyLogic(Logic):
    """A class representing fuzzy logic in torch."""

    def __init__(self, tnorm: TNorm = ProductTNorm(),
                 complement: Complement = StandardComplement(),
                 comparison: Comparison = SigmoidComparison(),
                 sampling: RandomSampling = SoftRandomSampling(),
                 rounding: Rounding = SoftRounding(),
                 control: ControlFlow = SoftControlFlow(),
                 eps: float = 1e-15,
                 use64bit: bool = False) -> None:
        super().__init__(use64bit=use64bit)
        self.tnorm = tnorm  # fuzzy AND
        self.complement = complement  # fuzzy NOT
        self.comparison = comparison  # relaxed comparisons
        self.sampling = sampling  # relaxed sampling for discrete RVs
        self.rounding = rounding  # relaxed rounding ops
        self.control = control  # relaxed control flow
        self.eps = eps  # underflow guard
    '''Creates a new fuzzy logic in jorch.
        
        :param tnorm: fuzzy operator for logical AND
        :param complement: fuzzy operator for logical NOT
        :param comparison: fuzzy operator for comparisons (>, >=, <, ==, ~=, ...)
        :param sampling: random sampling of non-reparameterizable distributions
        :param rounding: rounding floating values to integers
        :param control: if and switch control structures
        :param eps: small positive float to mitigate underflow
        :param use64bit: whether to perform arithmetic in 64 bit
        '''
    def __str__(self) -> str:
        return (f'model relaxation:\n'
                f'    tnorm        ={str(self.tnorm)}\n'
                f'    complement   ={str(self.complement)}\n'
                f'    comparison   ={str(self.comparison)}\n'
                f'    sampling     ={str(self.sampling)}\n'
                f'    rounding     ={str(self.rounding)}\n'
                f'    control      ={str(self.control)}\n'
                f'    underflow_tol={self.eps}\n'
                f'    use_64_bit   ={self.use64bit}\n')

    def summarize_hyperparameters(self) -> str:
        return self.__str__()

    # ===========================================================================
    # logical operators
    # ===========================================================================

    def logical_and(self, id, init_params):
        return self.tnorm.norm(id, init_params)

    def logical_not(self, id, init_params):
        return self.complement(id, init_params)

    def logical_or(self, id, init_params):
        _not1 = self.complement(f'{id}_~1', init_params)
        _not2 = self.complement(f'{id}_~2', init_params)
        _and = self.tnorm.norm(f'{id}_^', init_params)
        _not = self.complement(f'{id}_~', init_params)

        def _torch_wrapped_calc_or_approx(x, y, params):
            not_x, params = _not1(x, params)
            not_y, params = _not2(y, params)
            not_x_and_not_y, params = _and(not_x, not_y, params)  # De Morgan to build OR
            return _not(not_x_and_not_y, params)

        return _torch_wrapped_calc_or_approx

    def xor(self, id, init_params):
        _not = self.complement(f'{id}_~', init_params)
        _and1 = self.tnorm.norm(f'{id}_^1', init_params)
        _and2 = self.tnorm.norm(f'{id}_^2', init_params)
        _or = self.logical_or(f'{id}_|', init_params)

        def _torch_wrapped_calc_xor_approx(x, y, params):
            x_and_y, params = _and1(x, y, params)
            not_x_and_y, params = _not(x_and_y, params)
            x_or_y, params = _or(x, y, params)
            return _and2(x_or_y, not_x_and_y, params)

        return _torch_wrapped_calc_xor_approx

    def implies(self, id, init_params):
        _not = self.complement(f'{id}_~', init_params)
        _or = self.logical_or(f'{id}_|', init_params)

        def _torch_wrapped_calc_implies_approx(x, y, params):
            not_x, params = _not(x, params)
            return _or(not_x, y, params)

        return _torch_wrapped_calc_implies_approx

    def equiv(self, id, init_params):
        _implies1 = self.implies(f'{id}_=>1', init_params)
        _implies2 = self.implies(f'{id}_=>2', init_params)
        _and = self.tnorm.norm(f'{id}_^', init_params)

        def _torch_wrapped_calc_equiv_approx(x, y, params):
            x_implies_y, params = _implies1(x, y, params)
            y_implies_x, params = _implies2(y, x, params)
            return _and(x_implies_y, y_implies_x, params)

        return _torch_wrapped_calc_equiv_approx

    def forall(self, id, init_params):
        return self.tnorm.norms(id, init_params)

    def exists(self, id, init_params):
        _not1 = self.complement(f'{id}_~1', init_params)
        _not2 = self.complement(f'{id}_~2', init_params)
        _forall = self.forall(f'{id}_forall', init_params)

        def _torch_wrapped_calc_exists_approx(x, axis, params):
            not_x, params = _not1(x, params)
            forall_not_x, params = _forall(not_x, axis, params)  # exists = not forall not x
            return _not2(forall_not_x, params)

        return _torch_wrapped_calc_exists_approx

    # ===========================================================================
    # comparison operators
    # ===========================================================================

    def greater_equal(self, id, init_params):
        return self.comparison.greater_equal(id, init_params)

    def greater(self, id, init_params):
        return self.comparison.greater(id, init_params)

    def less_equal(self, id, init_params):
        _greater_eq = self.greater_equal(id, init_params)

        def _torch_wrapped_calc_leq_approx(x, y, params):
            return _greater_eq(-x, -y, params)  # reuse >= by flipping signs

        return _torch_wrapped_calc_leq_approx

    def less(self, id, init_params):
        _greater = self.greater(id, init_params)

        def _torch_wrapped_calc_less_approx(x, y, params):
            return _greater(-x, -y, params)  # reuse > by flipping signs

        return _torch_wrapped_calc_less_approx

    def equal(self, id, init_params):
        return self.comparison.equal(id, init_params)

    def not_equal(self, id, init_params):
        _not = self.complement(f'{id}_~', init_params)
        _equal = self.comparison.equal(f'{id}_==', init_params)

        def _torch_wrapped_calc_neq_approx(x, y, params):
            equal, params = _equal(x, y, params)
            return _not(equal, params)

        return _torch_wrapped_calc_neq_approx

    # ===========================================================================
    # special functions
    # ===========================================================================

    def sgn(self, id, init_params):
        return self.comparison.sgn(id, init_params)

    def floor(self, id, init_params):
        return self.rounding.floor(id, init_params)

    def round(self, id, init_params):
        return self.rounding.round(id, init_params)

    def ceil(self, id, init_params):
        _floor = self.rounding.floor(id, init_params)

        def _torch_wrapped_calc_ceil_approx(x, params):
            neg_floor, params = _floor(-x, params)
            return -neg_floor, params

        return _torch_wrapped_calc_ceil_approx

    def div(self, id, init_params):
        _floor = self.rounding.floor(id, init_params)

        def _torch_wrapped_calc_div_approx(x, y, params):
            return _floor(x / y, params)  # relaxed floor division

        return _torch_wrapped_calc_div_approx

    def mod(self, id, init_params):
        _div = self.div(id, init_params)

        def _torch_wrapped_calc_mod_approx(x, y, params):
            div, params = _div(x, y, params)
            return x - y * div, params  # x mod y using relaxed div

        return _torch_wrapped_calc_mod_approx

    def sqrt(self, id, init_params):
        def _torch_wrapped_calc_sqrt_approx(x, params):
            return torch.sqrt(x + self.eps), params  # add epsilon to avoid NaNs at 0

        return _torch_wrapped_calc_sqrt_approx

    # ===========================================================================
    # indexing
    # ===========================================================================

    def argmax(self, id, init_params):
        return self.comparison.argmax(id, init_params)

    def argmin(self, id, init_params):
        _argmax = self.argmax(id, init_params)

        def _torch_wrapped_calc_argmin_approx(x, axis, param):
            return _argmax(-x, axis, param)  # argmin(x) = argmax(-x)

        return _torch_wrapped_calc_argmin_approx

    # ===========================================================================
    # control flow
    # ===========================================================================

    def control_if(self, id, init_params):
        return self.control.if_then_else(id, init_params)

    def control_switch(self, id, init_params):
        return self.control.switch(id, init_params)

    # ===========================================================================
    # random variables
    # ===========================================================================

    def discrete(self, id, init_params):
        return self.sampling.discrete(id, init_params, self)

    def bernoulli(self, id, init_params):
        return self.sampling.bernoulli(id, init_params, self)

    def poisson(self, id, init_params):
        return self.sampling.poisson(id, init_params, self)

    def geometric(self, id, init_params):
        return self.sampling.geometric(id, init_params, self)

    def binomial(self, id, init_params):
        return self.sampling.binomial(id, init_params, self)

    def negative_binomial(self, id, init_params):
        return self.sampling.negative_binomial(id, init_params, self)
######################################################################################################3


logic = FuzzyLogic(comparison=SigmoidComparison(10000.0),
                   rounding=SoftRounding(10000.0),
                   control=SoftControlFlow(10000.0))


def _test_logical():
    print('testing logical')
    init_params = {}
    _and = logic.logical_and(0, init_params)
    _not = logic.logical_not(1, init_params)
    _gre = logic.greater(2, init_params)
    _or = logic.logical_or(3, init_params)
    _if = logic.control_if(4, init_params)
    print(init_params)

    # https://towardsdatascience.com/emulating-logical-gates-with-a-neural-network-75c229ec4cc9
    def test_logic(x1, x2, w):
        q1, w = _gre(x1, 0, w)
        q2, w = _gre(x2, 0, w)
        q3, w = _and(q1, q2, w)
        q4, w = _not(q1, w)
        q5, w = _not(q2, w)
        q6, w = _and(q4, q5, w)        
        cond, w = _or(q3, q6, w)
        pred, w = _if(cond, +1, -1, w)
        return pred
    
    x1 = torch.tensor([1, 1, -1, -1, 0.1, 15, -0.5], dtype=torch.float)
    x2 = torch.tensor([1, -1, 1, -1, 10, -30, 6], dtype=torch.float)
    print(test_logic(x1, x2, init_params))    


def _test_indexing():
    print('testing indexing')
    init_params = {}
    _argmax = logic.argmax(0, init_params)
    _argmin = logic.argmin(1, init_params)
    print(init_params)

    def argmaxmin(x, w):
        amax, w = _argmax(x, 0, w)
        amin, w = _argmin(x, 0, w)
        return amax, amin
        
    values = torch.tensor([2., 3., 5., 4.9, 4., 1., -1., -2.])
    amax, amin = argmaxmin(values, init_params)
    print(amax)
    print(amin)


def _test_control():
    print('testing control')
    init_params = {}
    _switch = logic.control_switch(0, init_params)
    print(init_params)
    
    pred = torch.linspace(0, 2, 10)
    case1 = torch.tensor([-10.] * 10)
    case2 = torch.tensor([1.5] * 10)
    case3 = torch.tensor([10.] * 10)
    cases = torch.stack([case1, case2, case3])
    switch, _ = _switch(pred, cases, init_params)
    print(switch)


def _test_random():
    print('testing random')
    key = torch.manual_seed(42)
    init_params = {}
    _bernoulli = logic.bernoulli(0, init_params)
    _discrete = logic.discrete(1, init_params)
    _geometric = logic.geometric(2, init_params)
    print(init_params)
    
    def bern(n, w):
        prob =torch.tensor([0.3] * n)
        sample, _ = _bernoulli(key, prob, w)
        return sample
    
    samples = bern(50000, init_params)
    print(torch.mean(samples))
    
    def disc(n, w):
        prob = torch.tensor([0.1, 0.4, 0.5])
        prob = torch.tile(prob, (n, 1))
        sample, _ = _discrete(key, prob, w)
        return sample
        
    samples = disc(50000, init_params)
    samples = torch.round(samples)
    print([torch.mean((samples == i).to(torch.float)) for i in range(3)])
    
    def geom(n, w):
        prob = torch.tensor([0.3] * n)
        sample, _ = _geometric(key, prob, w)
        return sample
    
    samples = geom(50000, init_params)
    print(torch.mean(samples))
    

def _test_rounding():
    print('testing rounding')
    init_params = {}
    _floor = logic.floor(0, init_params)
    _ceil = logic.ceil(1, init_params)
    _round = logic.round(2, init_params)
    _mod = logic.mod(3, init_params)
    print(init_params)
    
    x = torch.tensor([2.1, 0.6, 1.99, -2.01, -3.2, -0.1, -1.01, 23.01, -101.99, 200.01])
    print(_floor(x, init_params)[0])
    print(_ceil(x, init_params)[0])
    print(_round(x, init_params)[0])
    print(_mod(x, 2.0, init_params)[0])
    

if __name__ == '__main__':
    _test_logical()
    _test_indexing()
    _test_control()
    _test_random()
    _test_rounding()
    

# Reservoir / Water-Tank Problem Setup

## Scope and sources

This description is based on the current repository code, principally:

- `StochasticPBBP/problems/reservoir/domain.rddl`: RDDL variables, transition CPFs, reward, action preconditions, and state invariants.
- `StochasticPBBP/problems/reservoir/instance_3.rddl`: instance 3 objects, network, constants, initial state, horizon, and discount.
- `StochasticPBBP/problems/reservoir/instance_4.rddl`: instance 4 objects, network, constants, initial state, horizon, and discount.
- `StochasticPBBP/run.py:14-88`: experiment CLI, including the runtime horizon override and exact-evaluation setting.
- `StochasticPBBP/manager.py:33-96,424-550`: `ExperimentManager`, fuzzy training logic, policy/trainer construction, and exact evaluation.
- `StochasticPBBP/core/Compiler.py:316-524`: `TorchRDDLCompiler.compile` and `compile_transition`, which compile CPFs, reward, preconditions, and invariants.
- `StochasticPBBP/core/Rollout.py:23-40,43-176,245-361`: `RolloutTrace`, `TorchRolloutCell`, and `TorchRollout`.
- `StochasticPBBP/core/Train.py:19-78,204-233`: `Train`, which maximizes rollout return using RMSprop by minimizing negative return.
- `StochasticPBBP/utils/Policies.py:38-134,199-335`: `MBDPOPolicy.evaluate` and `NeuralStateFeedbackPolicy`.

No units for water level, flow, or time are specified in these files. The discussion therefore uses the code's unitless numerical quantities.

## 1. Reservoirs, capacities, initial levels, and desired ranges

`rlevel(?r)` is the state variable. `TOP_RES(?r)` is the operational upper cap used in the transition clamp and state invariant; it is therefore the code-level capacity. `MIN_LEVEL(?r)` and `MAX_LEVEL(?r)` are not hard capacity limits: they define the zero-penalty desired interval. The exact values are below.

### Instance 3 (`instance_3.rddl`)

Instance 3 contains 10 reservoirs (`t1`-`t10`; line 4).

| Tank | Initial `rlevel` | `MIN_LEVEL` | `MAX_LEVEL` | Capacity `TOP_RES` | Initial status |
|---|---:|---:|---:|---:|---|
| t1 | 174.29435671809895 | 99.28077123793895 | 242.39769006937888 | 365.1318947784891 | within desired range |
| t2 | 287.68862939282326 | 52.70500492913583 | 146.82518673502517 | 354.56815947704115 | above desired range |
| t3 | 103.86499851198514 | 15.816777351060948 | 65.0007902681157 | 159.98564094350357 | above desired range |
| t4 | 181.75042086357442 | 73.60735148065014 | 171.13792332611985 | 206.54329864865548 | above desired range |
| t5 | 125.04442249381736 | 49.43402235793882 | 163.53549045033566 | 328.8257122826851 | within desired range |
| t6 | 14.673884118330674 | 112.64249162782089 | 266.6245150000089 | 312.3631504108888 | below desired range |
| t7 | 103.22980823470064 | 19.655300953891356 | 45.29329435401057 | 128.04968259773747 | above desired range |
| t8 | 39.57577996245053 | 109.27118610302212 | 182.40872437041935 | 228.52315631505726 | below desired range |
| t9 | 118.27791170779733 | 55.454238538970834 | 212.3383176934107 | 316.604658365775 | within desired range |
| t10 | 85.09888760368129 | 81.34358088129179 | 113.89335852327719 | 124.32384546502664 | within desired range |

The values are defined at `instance_3.rddl:31-70` and `instance_3.rddl:76-86`. Initially, 4 tanks are inside the desired interval, 4 are above it, and 2 are below it.

### Instance 4 (`instance_4.rddl`)

Instance 4 contains 30 reservoirs (`t1`-`t30`; line 4).

| Tank | Initial `rlevel` | `MIN_LEVEL` | `MAX_LEVEL` | Capacity `TOP_RES` | Initial status |
|---|---:|---:|---:|---:|---|
| t1 | 310.9021409540228 | 307.8457616040964 | 352.08621589192586 | 370.2511536365 | within desired range |
| t2 | 263.2695653388287 | 170.5833286646818 | 222.4823455048396 | 314.2075301335589 | above desired range |
| t3 | 330.38026712423124 | 381.8152620332591 | 454.88786304299884 | 543.3714016103207 | below desired range |
| t4 | 84.0915571749488 | 92.37062682812075 | 125.42774704309568 | 196.7627044795634 | below desired range |
| t5 | 61.51294100659808 | 44.04088086709557 | 57.72843515596222 | 128.43792588999128 | above desired range |
| t6 | 190.7700437468991 | 61.25571458059964 | 101.53912819746859 | 336.4120433862389 | above desired range |
| t7 | 213.82450513432505 | 271.7644883582414 | 336.01129788142663 | 578.7412911980139 | below desired range |
| t8 | 103.91778624692176 | 195.29344528536754 | 237.60432372284092 | 246.63083713319503 | below desired range |
| t9 | 57.55115831091353 | 146.82516262380219 | 208.74547796405068 | 416.07841822494424 | below desired range |
| t10 | 128.32664868760077 | 34.94226774452913 | 60.15341869265288 | 163.3905153687553 | above desired range |
| t11 | 53.9981580341833 | 19.833008851973833 | 47.788776884028195 | 156.69189876210913 | above desired range |
| t12 | 179.32649767484207 | 200.5574027028445 | 282.2473502801705 | 431.75216738015325 | below desired range |
| t13 | 219.98418653408473 | 174.63866917309528 | 233.4307794582134 | 306.77505459609733 | within desired range |
| t14 | 243.6144586518739 | 71.30229552858087 | 142.89367679966534 | 411.17628049540457 | above desired range |
| t15 | 118.63369935405039 | 135.7383477467734 | 175.55547228158636 | 270.6627958071409 | below desired range |
| t16 | 18.07742750895943 | 146.90188055852474 | 218.8353672210235 | 400.30003389634334 | below desired range |
| t17 | 39.302958872181186 | 82.84231515660481 | 104.59647501199343 | 157.21354004225947 | below desired range |
| t18 | 119.78003000555769 | 155.93154499701973 | 182.06467763882677 | 188.38545993484576 | below desired range |
| t19 | 143.6456643753364 | 201.09909477800795 | 252.54986250424162 | 302.41024770456875 | below desired range |
| t20 | 378.21939108134904 | 357.3060634427003 | 412.3660190936405 | 423.91732447688514 | within desired range |
| t21 | 46.94730628618677 | 4.3576925396304365 | 25.18718393597787 | 117.07913163508672 | above desired range |
| t22 | 221.92379415490612 | 195.87312448813074 | 267.7299400733195 | 591.3174059784435 | within desired range |
| t23 | 306.4681185892686 | 201.81118180105202 | 269.21003887679245 | 542.8614141475184 | above desired range |
| t24 | 69.1722555272111 | 42.13045160880544 | 69.55728262022056 | 177.46515230239476 | within desired range |
| t25 | 207.18499204705358 | 178.78026784499477 | 246.55421681621934 | 364.4840185817551 | within desired range |
| t26 | 410.9698249408266 | 200.17853668313043 | 292.1487113173995 | 568.3288421102608 | above desired range |
| t27 | 95.47347310531676 | 162.54994042886054 | 233.550501447452 | 393.34096810663044 | below desired range |
| t28 | 178.65365067121317 | 93.42937583621682 | 142.5310467141272 | 350.720123865356 | above desired range |
| t29 | 74.74233181026413 | 5.474941757935831 | 33.09344806289597 | 159.22492112945372 | above desired range |
| t30 | 129.51566607964847 | 122.70063455310635 | 154.60937573559866 | 174.60888394270526 | within desired range |

The values are defined at `instance_4.rddl:144-263` and `instance_4.rddl:269-299`. Initially, 7 tanks are inside the desired interval, 11 are above it, and 12 are below it.

## 2. Inflows and network structure

Two terms add water to tank `r` (`domain.rddl:45-46,57-63`):

1. **Rain:** `rain(r) = abs(Normal(0, RAIN_VAR(r)))`. The domain comment calls `RAIN_VAR` the "Half normal variance parameter," and the custom Torch compiler explicitly treats the second argument as a variance, sampling `mean + sqrt(variance) * epsilon` (`Compiler.py:1711-1718`). Thus instance 3 has zero rain because every variance is `0.0`; instance 4 uses variance `50.0` (standard deviation `sqrt(50)` in the Torch training compiler) for every tank.
2. **Upstream releases:** `inflow(r) = sum_in RES_CONNECT(in,r) * individual_outflow(in)`. For a source tank `s`, `individual_outflow(s) = released_water(s) / (outdegree(s) + CONNECTED_TO_SEA(s))`. Thus its effective release is divided equally among all downstream connections and, when applicable, the sea outlet.

The exact upstream sets, derived directly from the listed `RES_CONNECT(source,target)` facts, are:

### Instance 3 upstream sets (`instance_3.rddl:7-29`)

- `t1: {}`; `t2: {t1}`; `t3: {t8,t5,t7}`; `t4: {t1,t2,t9}`; `t5: {t2,t9,t6}`.
- `t6: {}`; `t7: {t1,t2,t9,t4,t6,t5}`; `t8: {t9}`; `t9: {}`; `t10: {t2,t9,t6,t5,t7,t3}`.
- The only sea outlet is `t10` (`instance_3.rddl:30`). The graph has 23 directed reservoir-to-reservoir links.

### Instance 4 upstream sets (`instance_4.rddl:7-142`)

- `t1:{t17,t22,t23,t24,t7,t30}`; `t2:{t28,t29,t27}`; `t3:{}`; `t4:{t10,t17,t22,t30,t16,t8,t25}`; `t5:{t28}`.
- `t6:{t29,t23,t30,t1,t26}`; `t7:{t29,t27,t24}`; `t8:{t22,t24,t30,t18,t1,t26,t14,t19,t16}`; `t9:{t3,t10}`; `t10:{}`.
- `t11:{t3,t28,t29,t24,t7,t1,t26,t6,t19,t21,t8,t25,t15,t4}`; `t12:{t28,t22,t30,t26,t14}`; `t13:{t3,t23,t30}`; `t14:{t3,t9,t24}`; `t15:{t10,t27,t30,t13,t1,t26,t19,t21,t16,t8,t25}`.
- `t16:{t10,t9,t27,t24,t30,t26}`; `t17:{t3,t9}`; `t18:{t17,t23,t2}`; `t19:{t17,t27,t20,t22,t23,t1,t26}`; `t20:{t3,t17,t29}`.
- `t21:{t28,t5,t27,t7,t30,t1,t26,t14,t12}`; `t22:{t10,t27}`; `t23:{t28,t10,t22}`; `t24:{t3,t28,t22,t23}`; `t25:{t28,t10,t9,t27,t22,t23,t24,t1,t26,t14,t6,t19,t16,t8}`.
- `t26:{t5,t18,t13}`; `t27:{t28}`; `t28:{}`; `t29:{t3,t17}`; `t30:{t3,t10,t27,t22,t24}`.
- The only sea outlet is `t11` (`instance_4.rddl:143`). The graph has 136 directed reservoir-to-reservoir links.

## 3. Outflows, demand, evaporation, and overflow

- The agent requests `release(r)` for every reservoir (`domain.rddl:40-41`). The effective release is `released_water(r) = max(0, min(rlevel(r), release(r)))`, so the dynamics use neither a negative release nor more water than is currently present (`domain.rddl:51-52`).
- Effective release is subtracted from the source tank and redistributed equally to downstream tanks; the share assigned to the sea leaves the modeled reservoir system (`domain.rddl:57-63`).
- Evaporation is an uncontrolled loss: `evaporated(r) = EVAPORATION_FACTOR * rlevel(r) / TOP_RES(r)`. `EVAPORATION_FACTOR` defaults to `0.1` and neither instance overrides it (`domain.rddl:22,48-49`). The formula, rather than the comment, is the precise definition: at capacity it removes `0.1` units per step.
- `overflow(r) = max(0, rlevel(r) - released_water(r) - TOP_RES(r))` (`domain.rddl:54-55`). It uses the current level after release and does **not** include same-step rain or upstream inflow.
- There is no demand, consumption, service target, exogenous withdrawal, or release reward variable in the reservoir domain. The only modeled losses are controlled release to the sea and evaporation; the only other removal term is `overflow` as defined above.

## 4. Actions and neural policy

The action is one concurrent real-valued `release(r)` per tank. Its RDDL default is `0.0`; `max-nondef-actions = pos-inf`, so the instance does not limit how many reservoir releases may be non-default in one step (`domain.rddl:41`; instance 3 line 88; instance 4 line 301).

RDDL action preconditions require `0 <= release(r) <= TOP_RES(r)` for all tanks (`domain.rddl:76-79`). The neural controller is a fully connected state-feedback network built by `NeuralStateFeedbackPolicy` (`Policies.py:286-335`). It observes all state fluents because `TorchRolloutCell` falls back to `state_fluents` when there are no separate observation fluents (`Rollout.py:82-84,130-135`). For this domain, its input is the vector of all reservoir levels and its output is one raw release per reservoir. Hidden layers use `Tanh`; the output layer is linear (`Policies.py:303-320`).

Important implementation detail: `NeuralStateFeedbackPolicy._apply_action_constraints` currently returns the raw output unchanged, and action-space bound extraction is commented out (`Policies.py:212-231,268-283`). During the custom differentiable rollout, `TorchRDDLCompiler.compile_transition` computes precondition and invariant truth values and puts them in the transition log, but `TorchRolloutCell.step` only uses reward and termination; it does not penalize or terminate on failed preconditions/invariants (`Compiler.py:450-520`; `Rollout.py:137-155`). The dynamics still use the clipped `released_water` CPF. Exact evaluation is separately performed through `pyRDDLGym` in `MBDPOPolicy.evaluate` (`Policies.py:60-134`; `manager.py:501-538`). The behavior of `pyRDDLGym` on a raw action-precondition violation is external to this repository and is not restated here.

## 5. Dynamics

For each reservoir `r`, the exact RDDL update (`domain.rddl:45-63`) is:

```text
rain_t(r)       = abs(Normal(0, RAIN_VAR(r)))
evap_t(r)       = 0.1 * level_t(r) / TOP_RES(r)
released_t(r)   = max(0, min(level_t(r), action_t(r)))
overflow_t(r)   = max(0, level_t(r) - released_t(r) - TOP_RES(r))
share_t(r)      = released_t(r) / (outdegree(r) + sea_indicator(r))
inflow_t(r)     = sum_s RES_CONNECT(s,r) * share_t(s)
level_(t+1)(r)  = min(TOP_RES(r),
                      max(0, level_t(r) + inflow_t(r) + rain_t(r)
                             - evap_t(r) - overflow_t(r) - released_t(r)))
```

The update clamps every next level to `[0, TOP_RES(r)]`. Transfers are computed from source releases in the same step. The instance discount is `1.0`, so cumulative return is the undiscounted sum over the horizon (`instance_3.rddl:89-90`; `instance_4.rddl:302-303`; `RolloutTrace.return_` in `Rollout.py:36-40`).

The file-defined horizons are 120 for instance 3 and 2000 for instance 4. However, `run.py --horizon` is passed to `ExperimentManager`, which assigns `env.horizon = horizon` and constructs training rollouts with that value (`run.py:21,57-72`; `manager.py:48-50,83`). A command using `--horizon 120` therefore evaluates both instances for 120 steps, regardless of instance 4's file value.

## 6. Constraints

The complete explicit RDDL constraints are:

- **Action preconditions:** for every tank, `release(r) >= 0` and `release(r) <= TOP_RES(r)` (`domain.rddl:76-79`).
- **State invariants:** for every tank, `rlevel(r) >= 0` and `rlevel(r) <= TOP_RES(r)` (`domain.rddl:81-85`).
- **Concurrent actions:** releases for different reservoirs can be selected concurrently (`domain.rddl:3-8`), and the instances use `max-nondef-actions = pos-inf`.
- **Physical availability enforced in the CPF:** effective release is clipped to `[0, current rlevel]` (`domain.rddl:52`).
- **Next-state bounds enforced in the CPF:** the next level is clipped to `[0, TOP_RES]` (`domain.rddl:63`).

`MIN_LEVEL` and `MAX_LEVEL` are reward thresholds, not state invariants: states below `MIN_LEVEL` and above `MAX_LEVEL` are permitted but penalized.

There are no RDDL termination conditions in `domain.rddl`, so there is no explicit success state or failure state that ends an episode early.

## 7. Reward, objective, and penalties

The one-step reward is the sum over reservoirs of a piecewise penalty evaluated on the **next** level `rlevel'(r)` (`domain.rddl:66-73`). Neither instance overrides the default costs (`domain.rddl:25-27`):

```text
if MIN_LEVEL(r) <= next_level(r) <= MAX_LEVEL(r):
    penalty_r = 0
elif next_level(r) < MIN_LEVEL(r):
    penalty_r = -5 * (MIN_LEVEL(r) - next_level(r))
elif MAX_LEVEL(r) < next_level(r) <= TOP_RES(r):
    penalty_r = -10 * (next_level(r) - MAX_LEVEL(r))
else:
    penalty_r = -10 * (next_level(r) - MAX_LEVEL(r))
                -15 * overflow(r)
reward = sum_r penalty_r
```

The strict `< MIN_LEVEL` in the explanatory form follows from the preceding inclusive zero-penalty test: equality with `MIN_LEVEL` receives zero. Equality with `MAX_LEVEL` also receives zero.

There is no positive reward, direct release cost, demand-shortage cost, terminal bonus, or separate constraint-violation penalty. Each step's maximum exact reward is therefore `0`, achieved when every next level lies in its desired interval. `Train._run_training_batch` sums the rollout rewards, sets `loss = -objective`, and updates the policy with RMSprop, so training attempts to maximize cumulative reward (`Train.py:78,204-227`).

### Overflow-code caveat

Under the exact stated dynamics, the overflow penalty branch is unreachable. The next level is explicitly clamped so that `rlevel' <= TOP_RES`, and the reward's overflow branch is reached only after the preceding branch for `MAX_LEVEL < rlevel' <= TOP_RES` fails. In addition, when the state invariant holds, `overflow = max(0, current_level - released - TOP_RES)` is always zero because `current_level <= TOP_RES` and `released >= 0`. Same-step rain and upstream inflow that would exceed capacity are clipped away by the next-state `min` but are not included in the `overflow` fluent. This is a consequence of the equations as written, not an assumed physical interpretation.

Training uses `FuzzyLogic` with `ProductTNorm`, `SigmoidComparison`, `SoftControlFlow`, and other differentiable approximations (`manager.py:86-96`; classes in `core/Logic.py`). Consequently, intermediate training returns follow relaxed approximations to comparisons and conditionals. The logged exact evaluations use the separate `pyRDDLGym` environment (`manager.py:501-538`).

## 8. Good policies and undesirable behavior

A good policy, as defined solely by the code, maximizes undiscounted cumulative reward. Because all reward terms are non-positive, the ideal exact behavior is to bring and keep every reservoir's next level in its tank-specific inclusive interval `[MIN_LEVEL, MAX_LEVEL]` at every step, giving reward `0` per step. It must coordinate releases because one tank's release becomes downstream tanks' inflow, except for the share sent to the sea.

Undesirable behavior is any trajectory with next levels outside those desired intervals: low levels incur a slope-5 penalty per unit of deficit, while high levels incur a slope-10 penalty per unit of excess. Raw actions outside `[0, TOP_RES]` violate declared action preconditions, even though the effective-release CPF clips the quantity used by the dynamics. There is no explicit binary failure, early termination, demand violation, or success threshold in the domain.

## 9. Instance 3 versus instance 4

| Component | Instance 3 | Instance 4 |
|---|---|---|
| File | `problems/reservoir/instance_3.rddl` | `problems/reservoir/instance_4.rddl` |
| Internal RDDL name | `inst_reservoir_control_cont_3c` | `inst_reservoir_control_cont_5c` (the filename and internal suffix differ) |
| Reservoirs | 10 (`t1`-`t10`) | 30 (`t1`-`t30`) |
| Directed reservoir links | 23 | 136 |
| Reservoirs with no upstream link | `t1`, `t6`, `t9` | `t3`, `t10`, `t28` |
| Sea-connected reservoir | `t10` | `t11` |
| `RAIN_VAR` | `0.0` for all tanks | `50.0` for all tanks |
| File-defined horizon | 120 | 2000 |
| Discount | 1.0 | 1.0 |
| Initial desired-range status | 4 within, 4 above, 2 below | 7 within, 11 above, 12 below |
| Capacities, desired ranges, initial levels | Instance-specific values in the first table | Different instance-specific values in the second table |
| Runtime with `run.py --horizon 120` | 120 steps | 120 steps; CLI overrides the file's 2000 |

Instance 4 is not merely a longer version of instance 3: it has a different number of reservoirs, different directed network, different sea outlet, different capacity/range/initial-level values, and nonzero stochastic rainfall. The internal `5c` name in `instance_4.rddl` is an explicit naming inconsistency in the repository; this report does not infer why it exists.

## Report-ready paragraph

The reservoir-control problem is a concurrent continuous-action RDDL task in which an agent observes all reservoir levels and selects a release for every reservoir at each time step. Effective releases are clipped to the water currently available, divided equally among downstream reservoirs and any sea outlet, and combined with upstream transfers, nonnegative stochastic rainfall, and level-dependent evaporation. Each next water level is clamped between zero and its reservoir-specific `TOP_RES` capacity. The objective is to maximize undiscounted cumulative reward by maintaining every next level within its reservoir-specific `[MIN_LEVEL, MAX_LEVEL]` interval: deviations below the interval are penalized at 5 units per unit deficit and deviations above it at 10 units per unit excess. Instance 3 contains 10 reservoirs, 23 directed links, zero rainfall variability, and a 120-step file horizon; instance 4 contains 30 reservoirs, 136 directed links, `RAIN_VAR=50` for every tank, and a 2000-step file horizon, although the experiment CLI can override this horizon.

## Structured summary

- **State:** one real-valued `rlevel` per reservoir; all levels are observed by the neural policy.
- **Action:** one concurrent real-valued requested `release` per reservoir.
- **Exogenous input:** `abs(Normal(0, RAIN_VAR))`; zero-parameter in instance 3 and parameter 50 in instance 4.
- **Network input:** equal shares of effective releases from every upstream reservoir.
- **Losses:** effective release, evaporation `0.1 * level / capacity`, and the separately defined `overflow` term.
- **Transition:** mass-balance expression followed by clipping to `[0, TOP_RES]`.
- **Hard model bounds:** requested releases declared in `[0, TOP_RES]`; states declared and dynamically clamped in `[0, TOP_RES]`; effective release clipped to available water.
- **Desired operating range:** tank-specific inclusive `[MIN_LEVEL, MAX_LEVEL]`; this is a reward target, not a hard state bound.
- **Reward:** zero in-range; `-5` times low-level deficit; `-10` times high-level excess; an exact overflow branch exists textually but is unreachable under the exact clamped equations.
- **Objective:** maximize the sum of per-step rewards; discount is 1.0 and no terminal success/failure condition is defined.
- **Good policy:** coordinates releases to keep all next levels in their desired intervals as consistently as possible.
- **Undesirable behavior:** low/high desired-range violations and declared action-precondition violations; no other demand or failure criterion is encoded.

"""Minimal one-step examples for JAX compiler transition and torch simulator."""

#from __future__ import annotations

import copy
import sys
from pathlib import Path


import pyRDDLGym
from core.Policies import random_policy


PACKAGE_ROOT = Path(__file__).resolve().parent
print(f"PACKAGE_ROOT={PACKAGE_ROOT}")
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pyRDDLGym_jax.core.simulator import JaxRDDLSimulator  # noqa: E402
from core.Logic import ExactLogic  # noqa: E402
from core.Simulator import TorchRDDLSimulator  # noqa: E402
#reservoir

DOMAIN = PACKAGE_ROOT / "problems" / "reservoir" / "domain.rddl"
INSTANCE = PACKAGE_ROOT / "problems" / "reservoir" / "instance_1.rddl"
print(f"DOMAIN={DOMAIN}")
print(f"INSTANCE={INSTANCE}")


def torch_single_step(model, num_steps) -> None:
    print(f"=== TorchRDDLSimulator.step | {num_steps} steps ===")
    # here the simulator compiles the model 
    sim = TorchRDDLSimulator(model, logic=ExactLogic() , keep_tensors=True )#{"type": "smaller_1", "value": [2, 1]} 
    sim.seed(0)
    sim.reset()
    i = 0
    for _ in range(num_steps):
        i += 1
        rnd_policy = random_policy(model, logic=None)
        action = rnd_policy.get_action()
        obs, reward, done = sim.step(action, i)
        print(f"the step number{i} observation is {obs}")
        print(f"reward={float(reward)} done={done}")
    return num_steps


def main() -> None:
    #################################################################################
    ###### we can also run the torch simulator using the lifted model directly ######

    #from pyRDDLGym.core.parser.reader import RDDLReader
    #from pyRDDLGym.core.parser.parser import RDDLParser
    #from pyRDDLGym.core.compiler.model import RDDLLiftedModel
    #reader = RDDLReader(DOMAIN, INSTANCE)
    #domain = reader.rddltxt
    #parser = RDDLParser(lexer=None, verbose=False)
    #parser.build()
    #rddl = parser.parse(domain)
    #model_no_ptRDDLGym = RDDLLiftedModel(rddl)
    #torch_single_step(model_no_ptRDDLGym)
    #####################################################################################
    
    env = pyRDDLGym.make(domain=DOMAIN, instance=INSTANCE, vectorized=True)
    model = env.model
    torch_single_step(model , num_steps = 8)


    # print(np.array(obs_diff))
if __name__ == "__main__":
    main()

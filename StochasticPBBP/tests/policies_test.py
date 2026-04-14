


import sys
from pathlib import Path

import pyRDDLGym

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
print(f"####################PACKAGE_ROOT={PACKAGE_ROOT}")
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils.Policies import random_policy
from core.Compiler import TorchRDDLCompiler
from core.Logic import ExactLogic
from core.Initializer import RDDLValueInitializer


def main():
    root = Path(__file__).resolve().parents[1]
    print("#########3###########")
    print(f"PACKAGE_ROOT={root}")
    #reservoir
    domain = PACKAGE_ROOT / "problems" / "reservoir" / "domain.rddl"
    instance = PACKAGE_ROOT / "problems" / "reservoir" / "instance_1.rddl"
    env = pyRDDLGym.make(domain=domain, instance=instance, vectorized=True)
    model = env.model

    policy = random_policy(model, logic=None)

    action_template = policy.get_action_template()
    print(f"action template is {action_template}")

    action = policy.get_action()
    print(f"constant action is {action}")


if __name__ == "__main__":
    main()

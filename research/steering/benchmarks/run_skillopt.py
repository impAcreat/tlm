from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(os.environ.get("TLM_ROOT", Path(__file__).resolve().parents[3])).resolve()
SKILLOPT = Path(os.environ.get("SKILLOPT_ROOT", ROOT / "benchmarks/skillopt")).resolve()
for path in (ROOT, SKILLOPT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from research.steering.benchmarks.appworld import AppWorldAdapter
from research.steering.benchmarks.scienceworld import ScienceWorldAdapter


def main() -> None:
    spec = importlib.util.spec_from_file_location("skillopt_train_entry", SKILLOPT / "scripts/train.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module._ENV_REGISTRY.update({"scienceworld": ScienceWorldAdapter, "appworld": AppWorldAdapter})
    # The custom pilots do not need SkillOpt's optional benchmark adapters.
    # Skipping their eager imports avoids pulling every vision backend into an
    # otherwise text-only AppWorld/ScienceWorld run.
    module._register_builtins = lambda: None
    module.main()


if __name__ == "__main__":
    main()

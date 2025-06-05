from pathlib import Path
import sys

# Ensure repository root is on sys.path so package imports work
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Re-export top level modules
import back as _back
import front as _front
import orchestrator as _orchestrator

sys.modules[__name__ + '.back'] = _back
sys.modules[__name__ + '.front'] = _front
sys.modules[__name__ + '.orchestrator'] = _orchestrator

__all__ = ["back", "front", "orchestrator"]

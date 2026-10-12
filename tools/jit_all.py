"""jit.json over every tensorfold module (capture.py's specialization(), all modules imported first)."""
import importlib, json, pkgutil, sys
sys.path.insert(0, sys.argv[1])
import tensorfold
for m in pkgutil.walk_packages(tensorfold.__path__, "tensorfold."):
    if ".cuda" in m.name or "drafters" in m.name:
        try: importlib.import_module(m.name)
        except Exception as e: print("skip", m.name, type(e).__name__, file=sys.stderr)
from capture import specialization
open(sys.argv[2], "w").write(json.dumps(specialization(), indent=1, sort_keys=True) + "\n")
print(len(specialization()), "functions")

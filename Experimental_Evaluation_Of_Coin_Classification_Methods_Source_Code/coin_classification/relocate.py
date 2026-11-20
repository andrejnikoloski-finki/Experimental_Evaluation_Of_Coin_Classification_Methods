import json
from pathlib import Path


root = Path(__file__).resolve().parent
p = root / "experiments_28" / "protocol.json"
q = json.loads(p.read_text(encoding="utf-8"))

if (p.parent / "runs").exists():
    raise RuntimeError("Refusing to relocate after training has started")

q["dataset_root"] = str((root / "dataset_types").resolve())
tmp = p.with_suffix(".json.tmp")
tmp.write_text(json.dumps(q, indent=2, ensure_ascii=False), encoding="utf-8")
tmp.replace(p)
print("Dataset root:", q["dataset_root"])

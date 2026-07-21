# Local Smoke Tests

Use these checks before running a full Medea workflow. They avoid downloading
MedeaDB or calling an LLM, so they are suitable for local setup and CI.

## Package Import

```bash
python -c "import medea; print('medea import ok')"
```

This confirms the package is importable from the active environment.

## Syntax Check

```bash
python -m compileall -q medea main.py setup.py
```

This catches syntax errors in the main package and top-level entrypoints without
requiring credentials.

## Configuration Check

```bash
cp env_template.txt .env
python - <<'PY'
from pathlib import Path

required = {"MEDEADB_PATH", "BACKBONE_LLM"}
keys = set()
for line in Path(".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        keys.add(line.split("=", 1)[0].strip())

missing = sorted(required - keys)
if missing:
    raise SystemExit(f"Missing required keys: {missing}")
print("configuration keys present")
PY
```

After these checks pass, download MedeaDB and configure a model provider before
running the full examples.

"""Console entry points for Medea."""

from pathlib import Path
import runpy


def main():
    """Run the repository evaluation CLI used by ``python main.py``."""
    entrypoint = Path(__file__).resolve().parents[1] / "main.py"
    if not entrypoint.exists():
        raise FileNotFoundError(
            "Could not find main.py next to the Medea package. "
            "Run `python main.py` from a source checkout instead."
        )
    runpy.run_path(str(entrypoint), run_name="__main__")

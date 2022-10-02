"""Training paths."""
from pathlib import Path

TRAINING_DIR = Path(__file__).parents[0].resolve()
ARTIFACTS_DIR = TRAINING_DIR.parent / "text_recognizer" / "artifacts"
RUNS_DIR = TRAINING_DIR / "logs" / "runs"

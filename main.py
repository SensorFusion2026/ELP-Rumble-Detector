"""Compatibility launcher for the package-based training script."""

import sys
from pathlib import Path

# Allow `python main.py` from a fresh clone without requiring editable install.
repo_root = Path(__file__).resolve().parent
src_dir = repo_root / "src"
if src_dir.is_dir():
    sys.path.insert(0, str(src_dir))

from elp_rumble.training.train_compare import main


if __name__ == "__main__":
    main()

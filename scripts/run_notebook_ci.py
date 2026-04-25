#!/usr/bin/env python3
"""CI entrypoint: pytest + offline async smoke (no API keys)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pytest = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=_ROOT,
        check=False,
    )
    if pytest.returncode != 0:
        sys.exit(pytest.returncode)

    smoke = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "smoke_evaluation.py")],
        cwd=_ROOT,
        check=False,
    )
    sys.exit(smoke.returncode)


if __name__ == "__main__":
    main()

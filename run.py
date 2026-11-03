#!/usr/bin/env python3
"""
Entrypoint na raiz do repositório que delega para `ai/run.py`.
Permite executar `python run.py --"ai/config.yaml"` ou `python run.py --config ai/config.yaml`
diretamente a partir da raiz `/home/jalcantara/DeepLearning2026-3`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.run import main


if __name__ == "__main__":
    raise SystemExit(main())


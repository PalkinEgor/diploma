"""
Использование:
    python scripts/run.py --config configs/<...>.yaml
"""

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from proto_tokens.experiment import run_experiment
from proto_tokens.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Запуск эксперимента proto_tokens по YAML-конфигу.")
    parser.add_argument("--config", required=True, help="Путь к YAML-конфигу эксперимента.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()

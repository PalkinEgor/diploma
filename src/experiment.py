from . import analysis  # (регистрация noise/attention)
from . import models  # (регистрация энкодеров)
from . import training  # (регистрация тренеров/оптимизатора/teacher)
from .data import generation  # (регистрация генераторов данных)
from .registry import EXPERIMENTS
from .utils import fix_seeds


def run_experiment(config: dict):
    """Собрать и запустить эксперимент, описанный конфигом.

    Требуется поле ``experiment.type``; остальные разделы конфига интерпретирует
    конкретный ``Runnable``.
    """
    exp_cfg = config.get("experiment", {})
    exp_type = exp_cfg.get("type")
    if exp_type is None:
        raise KeyError(
            "В конфиге нет 'experiment.type'. Доступные типы: "
            f"{EXPERIMENTS.available()}"
        )

    fix_seeds(config.get("training", {}).get("seed"))

    runnable_cls = EXPERIMENTS.get(exp_type)
    runnable = runnable_cls.from_config(config)
    return runnable.run()

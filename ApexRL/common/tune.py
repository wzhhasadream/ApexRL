from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import optuna


def tune(
    objective: Callable[[optuna.Trial], float],
    n_trials: int = 20,
    *,
    sampler: optuna.samplers.BaseSampler | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
    seed: int = 1,
    direction: Literal["maximize", "minimize"] = "maximize",
    study_name: str | None = None,
    storage: str | optuna.storages.BaseStorage | None = None,
    storage_file: str | None = None,   # JournalStorage file path; takes precedence over storage
    timeout: float | None = None,
) -> optuna.Study:
    if storage_file is not None:
        storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(storage_file))
    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction=direction, sampler=sampler, pruner=pruner,
        study_name=study_name, storage=storage,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study

"""MLflow helper utilities."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import mlflow


def init_mlflow(experiment_name: str | None = None, tracking_uri: str | None = None) -> None:
    """Initialise MLflow experiment."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment_name:
        mlflow.set_experiment(experiment_name)


@contextmanager
def mlflow_run(run_name: Optional[str], params: Dict[str, object] | None = None) -> Iterator[mlflow.ActiveRun | None]:
    """Context manager that starts and ends an MLflow run."""
    if run_name is None:
        yield None
        return
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        yield run

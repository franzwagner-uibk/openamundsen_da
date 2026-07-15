"""
openamundsen_da
Author: Franz Wagner
Date: 2025-10-30
"""

from openamundsen_da.exceptions import (
    CleanupSafetyError,
    ObservationPreprocessingError,
    OpenAmundsenDAError,
    ProjectCleanupError,
    ProjectPreparationError,
    ProjectRenderError,
    ProjectRunError,
    ProjectValidationError,
)
from openamundsen_da.results import (
    CleanupFailure,
    CleanupResult,
    ObservationPreprocessingResult,
    ObservationProduct,
    PreparationResult,
    RenderResult,
    RunResult,
    WorkflowStatus,
)

__version__ = "1.0.0"


def __getattr__(name: str):
    if name in {"clean_project", "prepare_project", "render_project", "run_project"}:
        from openamundsen_da import api

        return getattr(api, name)
    if name in {"preprocess_snow_cover", "preprocess_wet_snow"}:
        from openamundsen_da import observations

        return getattr(observations, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CleanupFailure",
    "CleanupResult",
    "CleanupSafetyError",
    "ObservationPreprocessingError",
    "ObservationPreprocessingResult",
    "ObservationProduct",
    "OpenAmundsenDAError",
    "PreparationResult",
    "ProjectCleanupError",
    "ProjectPreparationError",
    "ProjectRenderError",
    "ProjectRunError",
    "ProjectValidationError",
    "RenderResult",
    "RunResult",
    "WorkflowStatus",
    "clean_project",
    "prepare_project",
    "preprocess_snow_cover",
    "preprocess_wet_snow",
    "render_project",
    "run_project",
]

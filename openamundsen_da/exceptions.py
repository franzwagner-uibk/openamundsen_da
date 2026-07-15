"""Public exception hierarchy for openAMUNDSEN-DA workflows."""

from __future__ import annotations


class OpenAmundsenDAError(Exception):
    """Base class for supported public workflow failures."""


class ProjectValidationError(OpenAmundsenDAError):
    """Raised when a setup/project contract is invalid."""

    def __init__(self, errors: str | list[str] | tuple[str, ...]) -> None:
        if isinstance(errors, str):
            errors = [errors]
        self.errors = tuple(str(error) for error in errors)
        super().__init__("Project validation failed:\n- " + "\n- ".join(self.errors))


class ObservationPreprocessingError(OpenAmundsenDAError):
    """Raised when observation preprocessing fails."""


class ProjectPreparationError(OpenAmundsenDAError):
    """Raised when deterministic project preparation fails."""


class ProjectRunError(OpenAmundsenDAError):
    """Raised when project execution or required output validation fails."""


class ProjectRenderError(OpenAmundsenDAError):
    """Raised when configured project rendering fails."""


class ProjectCleanupError(OpenAmundsenDAError):
    """Raised when project cleanup cannot be completed."""


class CleanupSafetyError(ProjectCleanupError):
    """Raised when a cleanup candidate is outside the owned artifact set."""


__all__ = [
    "CleanupSafetyError",
    "ObservationPreprocessingError",
    "OpenAmundsenDAError",
    "ProjectCleanupError",
    "ProjectPreparationError",
    "ProjectRenderError",
    "ProjectRunError",
    "ProjectValidationError",
]

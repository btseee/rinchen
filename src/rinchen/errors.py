"""Domain exceptions for Rinchen, per <coding_standards> in the master plan."""


class RinchenError(Exception):
    """Base class for all Rinchen domain exceptions."""


class ConfigError(RinchenError):
    """Raised for invalid or missing configuration."""


class DataError(RinchenError):
    """Raised for corrupt, missing, or invalid data."""


class UpstreamError(RinchenError):
    """Raised when an upstream dependency (Eole, CTranslate2, etc.) misbehaves."""


class ParityError(RinchenError):
    """Raised when a fallback implementation fails to match its reference within tolerance."""


class DoctorCheckFailed(RinchenError):
    """Raised by a `rinchen doctor` check that fails."""

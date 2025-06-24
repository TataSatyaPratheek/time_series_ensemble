"""Custom exceptions for the time series ensemble project."""

class TimeSeriesEnsembleError(Exception):
    """Base exception for time series ensemble errors."""
    pass

class DataLoadError(TimeSeriesEnsembleError):
    """Exception raised when data loading fails."""
    pass

class ValidationError(TimeSeriesEnsembleError):
    """Exception raised when data validation fails."""
    pass

class PreprocessingError(TimeSeriesEnsembleError):
    """Exception raised when preprocessing fails."""
    pass

class ModelError(TimeSeriesEnsembleError):
    """Exception raised when model operations fail."""
    pass

class AgentError(TimeSeriesEnsembleError):
    """Exception raised when agent operations fail."""
    pass

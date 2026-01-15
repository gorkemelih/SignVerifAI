"""Path resolution utilities for SignVerify."""

from pathlib import Path
from typing import Union


def resolve_path(path: Union[str, Path], base: Path) -> Path:
    """
    Resolve a path relative to a base directory.
    
    Args:
        path: Path to resolve (can be absolute or relative)
        base: Base directory for relative paths
    
    Returns:
        Resolved absolute path
    """
    path = Path(path)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def ensure_parent_exists(path: Path) -> Path:
    """Ensure parent directory exists and return path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_relative_path(path: Path, base: Path) -> str:
    """Get path relative to base directory as string."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)

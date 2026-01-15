"""I/O utilities for SignVerify.

Provides helpers for reading/writing CSV, JSON, and other file formats.
"""

import json
from pathlib import Path
from typing import Any, Union

import pandas as pd


def read_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Read CSV file into DataFrame."""
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Union[str, Path], index: bool = False) -> None:
    """Write DataFrame to CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def read_json(path: Union[str, Path]) -> dict:
    """Read JSON file into dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict, path: Union[str, Path], indent: int = 2) -> None:
    """Write dict to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def write_markdown(content: str, path: Union[str, Path]) -> None:
    """Write content to Markdown file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def read_text(path: Union[str, Path]) -> str:
    """Read text file content."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

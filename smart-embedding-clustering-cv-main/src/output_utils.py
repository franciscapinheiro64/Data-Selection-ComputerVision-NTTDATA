from datetime import datetime
from pathlib import Path

from src.config import (
    RESULTS_DATE_FORMAT,
    RESULTS_DIR,
)


def get_output_dir(
    base_dir: str | Path = RESULTS_DIR,
    date: datetime | None = None,
) -> Path:
    """
    Create and return the directory where output files are saved.

    Args:
        base_dir (str | pathlib.Path): Base directory used to store the
            generated results.
        date (datetime | None): Date used to create the output folder.
            If None, the current date is used.

    Returns:
        pathlib.Path: Path to the output directory.
    """
    base_dir = Path(base_dir)

    current_date = date or datetime.now()

    folder_name = current_date.strftime(
        RESULTS_DATE_FORMAT,
    )

    output_dir = base_dir / folder_name

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir

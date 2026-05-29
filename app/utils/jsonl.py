import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

JsonRecord = Dict[str, Any]


def write_jsonl(
    path: str,
    rows: Iterable[JsonRecord],
) -> None:
    """
    Write records to a JSONL file.

    Each record is written as a single JSON object
    on its own line.

    Args:
        path: Output JSONL file path.
        rows: Collection of JSON-serializable records.
    """
    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        for row in rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
            )
            file.write("\n")


def read_jsonl(
    path: str,
) -> List[JsonRecord]:
    """
    Read records from a JSONL file.

    Args:
        path: JSONL file path.

    Returns:
        List of parsed JSON objects.
    """
    input_path = Path(path)

    if not input_path.exists():
        return []

    with input_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]
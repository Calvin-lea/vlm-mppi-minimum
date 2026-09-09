import csv
import json
import math
from pathlib import Path


def _serializable(record):
    result = {}
    for key, value in record.items():
        if isinstance(value, float) and not math.isfinite(value):
            result[key] = None
        else:
            result[key] = value
    return result


def write_records(records, output_directory):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    cleaned = [_serializable(record) for record in records]
    jsonl_path = output / "episodes.jsonl"
    csv_path = output / "episodes.csv"
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for record in cleaned:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    if cleaned:
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(cleaned[0].keys()))
            writer.writeheader()
            writer.writerows(cleaned)
    return str(csv_path), str(jsonl_path)


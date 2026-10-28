"""Inspect recovery progress and schema without deserializing model weights."""

import argparse
import json
import zipfile
from pathlib import Path


def inspect_recovery(path, *, include_schema=False):
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        metadata["artifacts"] = archive.namelist()
    if not include_schema:
        metadata["state_leaves"] = len(metadata.pop("schema"))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--schema", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(inspect_recovery(args.archive, include_schema=args.schema), indent=2)
    )


if __name__ == "__main__":
    main()

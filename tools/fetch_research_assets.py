"""Fetch only the pinned LEAP meshes; preview unless --run is specified."""

import argparse
import subprocess
from pathlib import Path

from mujoco_playground._src import mjx_env

REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie.git"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=Path("data/assets/mujoco_menagerie")
    )
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    commit = mjx_env.MENAGERIE_COMMIT_SHA
    print(f"{'Fetch' if args.run else 'DRY RUN'}: {REPOSITORY}@{commit}")
    print(f"LEAP assets: {args.destination}/leap_hand/assets")
    if not args.run:
        return
    if args.destination.exists():
        raise FileExistsError("Choose a fresh asset destination")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    # A narrow checkout avoids downloading unrelated robot assets or using siblings.
    subprocess.run(
        ["git", "init", str(args.destination)], check=True, capture_output=True
    )
    prefix = ["git", "-C", str(args.destination)]
    subprocess.run([*prefix, "remote", "add", "origin", REPOSITORY], check=True)
    subprocess.run([*prefix, "sparse-checkout", "set", "leap_hand"], check=True)
    subprocess.run(
        [*prefix, "fetch", "--depth=1", "--filter=blob:none", "origin", commit],
        check=True,
    )
    subprocess.run([*prefix, "checkout", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output([*prefix, "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise ValueError("Fetched Menagerie commit differs from Playground's pin")


if __name__ == "__main__":
    main()

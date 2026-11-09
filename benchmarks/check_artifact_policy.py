"""Reject new raw experiment data and large blobs in a Git change."""

import argparse
import re
import subprocess
from pathlib import PurePosixPath

LIMIT = 512 * 1024
RAW_SUFFIXES = {".npz", ".npy", ".off", ".vtu", ".gz", ".zip", ".tar", ".sqlite", ".nsys-rep"}


def violation(name, content):
    path = PurePosixPath(name)
    if len(content) > LIMIT:
        return "blob exceeds 512 KiB; publish a release artifact or LFS pointer"
    if name.startswith((".work/", "results/local/")):
        return "local experiment path must remain ignored"
    if name.startswith("tests/data/"):
        return None  # Small deterministic regression fixtures are deliberate checkout inputs.
    if path.suffix in RAW_SUFFIXES or b"\0" in content:
        if re.fullmatch(
            rb"version https://git-lfs.github.com/spec/v1\noid sha256:[0-9a-f]{64}\nsize [0-9]+\n",
            content,
        ):
            return None
        return "raw/binary artifact belongs in a release bundle or LFS"
    return None


def check(base, head="HEAD"):
    # Resolve arguments first: subsequent Git commands receive only object IDs.
    def resolve(ref):
        return subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"], text=True
        ).strip()

    base, head = resolve(base), resolve(head)
    names = subprocess.check_output(
        ["git", "diff", "--name-only", "-z", "--no-renames", "--diff-filter=AM", base, head, "--"]
    )
    failures = []
    for name in names.decode().split("\0"):
        if not name:
            continue
        obj = head + ":" + name
        size = int(subprocess.check_output(["git", "cat-file", "-s", obj]))
        # Avoid reading an oversized blob into Python; Git may fetch it for its size.
        problem = (
            "blob exceeds 512 KiB; use release artifacts or LFS"
            if size > LIMIT
            else violation(name, subprocess.check_output(["git", "cat-file", "blob", obj]))
        )
        if problem:
            failures.append((name, problem))
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    failures = check(args.base, args.head)
    for name, reason in failures:
        print(f"{name}: {reason}")
    if failures:
        raise SystemExit(1)
    print("Artifact policy passed for changed files; historical blobs were not migrated.")


if __name__ == "__main__":
    main()

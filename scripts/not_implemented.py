"""Fail clearly for Makefile targets whose code lands in a later block."""

import sys


def main() -> int:
    """Print which block implements the target and exit non-zero."""
    target, block = sys.argv[1], sys.argv[2]
    print(f"make {target}: not implemented yet (arrives in {block}).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

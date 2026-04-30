from __future__ import annotations

from . import cli_main


def main() -> int:
    return cli_main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

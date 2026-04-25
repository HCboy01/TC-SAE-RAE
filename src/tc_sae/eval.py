#!/usr/bin/env python3
"""TC-SAE evaluation entrypoint.

Standalone evaluation is not separated from `train.py` yet.
For now, validation metrics are computed during training.
"""


def main() -> None:
    raise SystemExit(
        "Standalone TC-SAE evaluation is not implemented yet. "
        "Use src/tc_sae/train.py for training-time validation."
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create a runnable release config with user-provided paths.")
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-h5", required=True, type=str)
    parser.add_argument("--test-h5", required=True, type=str)
    parser.add_argument("--train-txt", required=True, type=str)
    parser.add_argument("--test-txt", required=True, type=str)
    args = parser.parse_args()

    config = json.loads(args.base_config.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config.update(
        {
            "model_dir": str((output_dir / "model").resolve()),
            "record_dir": str((output_dir / "record").resolve()),
            "log_dir": str((output_dir / "log").resolve()),
            "log_txt": str((output_dir / "log.txt").resolve()),
            "h5_file": args.train_h5,
            "test_h5_file": args.test_h5,
            "train_txt_file": str(Path(args.train_txt).resolve()),
            "test_txt_file": str(Path(args.test_txt).resolve()),
        }
    )

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
    print(f"config: {args.output_config}")


if __name__ == "__main__":
    main()

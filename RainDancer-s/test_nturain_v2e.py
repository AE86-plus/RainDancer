#!/usr/bin/env python3
from pathlib import Path

import torch

from infer_bgr_rgb_sliding_common import build_parser, run_one_dataset


REPO_ROOT = Path(__file__).resolve().parent

DATASET_SPEC = {
    "name": "NTURain-v2e_test",
    "type": "legacy_paired",
    "config": REPO_ROOT / "options" / "v1.json",
    "checkpoint": REPO_ROOT / "weights" / "nturain" / "best.pth.tar",
}

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "inference" / "nturain_v2e"
TITLE = "RainDancer-s test on NTURain-v2e (sliding 128 stride 64)"


def main():
    parser = build_parser(TITLE, DEFAULT_OUTPUT_ROOT, 64)
    parser.add_argument("--config", type=str, default=str(DATASET_SPEC["config"]))
    parser.add_argument("--checkpoint", type=str, default=str(DATASET_SPEC["checkpoint"]))
    parser.add_argument("--test-h5", type=str, required=True, help="Path to the NTURain-v2e test H5 file.")
    parser.add_argument("--dataset-name", type=str, default=DATASET_SPEC["name"])
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    spec = {
        "name": args.dataset_name,
        "type": DATASET_SPEC["type"],
        "config": Path(args.config),
        "checkpoint": Path(args.checkpoint),
        "h5": args.test_h5,
    }

    print("=" * 100)
    print(TITLE)
    print(f"Output Root : {output_root}")
    print(f"Crop Size   : {args.crop_size}")
    print(f"Stride      : {args.stride}")
    print(f"Patch Batch : {args.patch_batch}")
    print("=" * 100)

    run_one_dataset(spec, args, output_root)
    print("\nFinished.")
    print(f"Results root: {output_root}")


if __name__ == "__main__":
    main()

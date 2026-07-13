#!/usr/bin/env python3

import argparse
from pathlib import Path

import h5py

def natural_scene_keys(group):
    def key_fn(value):
        value = str(value)
        if value.isdigit():
            return (0, int(value))
        if value.startswith("seq_"):
            tail = value[4:]
            if tail.isdigit():
                return (1, int(tail))
            if tail.startswith("long_") and tail[5:].isdigit():
                return (2, int(tail[5:]))
        return (3, value)
    return sorted(group.keys(), key=key_fn)

def build_entries(h5_path, sequence_length):
    with h5py.File(h5_path, "r") as h5f:
        entries = []
        if {"input", "processed"}.issubset(h5f.keys()):
            for scene in natural_scene_keys(h5f["input"]):
                frame_keys = sorted(k for k in h5f["input"][scene].keys() if k != "voxel")
                if len(frame_keys) < sequence_length:
                    continue
                max_start = len(frame_keys) - sequence_length + 1
                for start in range(max_start):
                    entries.append(f"input/{scene}/{start:05d}")
        elif {"rainy", "gt"}.issubset(h5f.keys()):
            for scene in natural_scene_keys(h5f["rainy"]):
                frame_keys = sorted(k for k in h5f["rainy"][scene].keys() if k != "voxel")
                if len(frame_keys) < sequence_length:
                    continue
                max_start = len(frame_keys) - sequence_length + 1
                for start in range(max_start):
                    entries.append(f"rainy/{scene}/{start:05d}")
        else:
            raise ValueError(f"{h5_path} is not a supported RainSyn/RainVIDSS-style h5 file")
        return entries

def write_entries(entries, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(entries) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate RainSyn/RainVIDSS train/test txt files for Our-style projects.")
    parser.add_argument("--train-h5", required=True, type=Path)
    parser.add_argument("--test-h5", required=True, type=Path)
    parser.add_argument("--train-out", required=True, type=Path)
    parser.add_argument("--test-out", required=True, type=Path)
    parser.add_argument("--sequence-length", default=3, type=int)
    args = parser.parse_args()

    train_entries = build_entries(args.train_h5, args.sequence_length)
    test_entries = build_entries(args.test_h5, args.sequence_length)

    write_entries(train_entries, args.train_out)
    write_entries(test_entries, args.test_out)

    print(f"train entries: {len(train_entries)} -> {args.train_out}")
    print(f"test entries:  {len(test_entries)} -> {args.test_out}")

if __name__ == "__main__":
    main()

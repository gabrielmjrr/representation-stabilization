"""
Create the directory tree for a fullstudy run before training begins.

Usage:
    python setup_storage.py configs/resnet18_cifar10_200_fullstudy.yaml

All required directories are read from the config's paths section, so the
config is the single source of truth for storage layout.
"""

import sys
import os
import yaml


def create_directories(config_path: str) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    paths_section = config.get("paths")
    if paths_section is None:
        raise ValueError(f"Config {config_path} has no 'paths' section.")

    print(f"Setting up storage for config: {config_path}")
    print(f"Base directory: {paths_section['base']}")
    print()

    for key, directory in paths_section.items():
        if key == "base":
            # base is a parent; also create it
            os.makedirs(directory, exist_ok=True)
            print(f"  [base]        {directory}")
            continue
        os.makedirs(directory, exist_ok=True)
        print(f"  [{key:<12}] {directory}")

    print()
    print("All directories created (or already exist).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python setup_storage.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    if not os.path.isfile(config_path):
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)

    create_directories(config_path)

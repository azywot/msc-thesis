"""Dataset downloader utility.

This script helps download and prepare datasets for use with agent_engine.
"""

import argparse
from pathlib import Path

DATASETS_INFO = {
    "gaia": {
        "description": "GAIA (General AI Assistant) benchmark",
        "url": "https://huggingface.co/datasets/gaia-benchmark/GAIA",
        "format": "JSONL",
        "location": "data/GAIA/",
    },
    "gpqa": {
        "description": "GPQA (Graduate-level Physics Questions)",
        "url": "https://huggingface.co/datasets/Idavidrein/gpqa",
        "format": "JSONL",
        "location": "data/GPQA/",
    },
    "math500": {
        "description": "MATH500 challenging math problems",
        "url": "https://github.com/hendrycks/math",
        "format": "JSONL",
        "location": "data/MATH500/",
    },
}


def list_datasets():
    """List available datasets."""
    print("Available datasets:")
    print("="*60)
    for name, info in DATASETS_INFO.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  URL: {info['url']}")
        print(f"  Location: {info['location']}")


def download_dataset(name: str, data_dir: Path):
    """Download a dataset.

    Args:
        name: Dataset name
        data_dir: Data directory
    """
    if name not in DATASETS_INFO:
        print(f"Error: Unknown dataset '{name}'")
        print(f"Available datasets: {', '.join(DATASETS_INFO.keys())}")
        return

    info = DATASETS_INFO[name]
    target_dir = data_dir / name.upper()
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {name}")
    print(f"Description: {info['description']}")
    print(f"Target directory: {target_dir}")
    print(f"\nPlease download the dataset from:")
    print(f"  {info['url']}")
    print(f"\nAnd place the files in:")
    print(f"  {target_dir}")
    print(f"\nExpected format: {info['format']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download datasets for agent_engine")
    parser.add_argument("--list", action="store_true",
                       help="List available datasets")
    parser.add_argument("--dataset", type=str,
                       help="Dataset name to download")
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="Data directory (default: ./data)")
    args = parser.parse_args()

    if args.list:
        list_datasets()
    elif args.dataset:
        download_dataset(args.dataset, Path(args.data_dir))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""Dataset downloader utility.

Downloads and converts all benchmark datasets to JSONL format.
HF token is optional — only GAIA, GPQA, and HLE (gated) require it.
"""

import argparse
import base64
import json
import os
import random
import sys

from datasets import load_dataset, DownloadConfig
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from agent_engine.utils import set_seed, get_seed_from_env

def _download_cfg(token: str | None) -> DownloadConfig | None:
    return DownloadConfig(token=token) if token else None


def _warn_token(dataset_name: str):
    print(
        f"Warning: HF_TOKEN not set — {dataset_name} requires gated access. "
        "Proceed only if you have accepted the dataset terms on HuggingFace.",
        file=sys.stderr,
    )


def _save_jsonl(data: list[dict], output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(data)} examples to {out_path}")


def _apply_subset(data: list, subset: int, seed: int = 0) -> list:
    """Shuffle and truncate data to `subset` examples. No-op if subset <= 0.

    Always re-seeds before shuffling so the selected subset is deterministic
    regardless of any random calls made during data loading.
    """
    if subset > 0 and subset < len(data):
        set_seed(seed)
        random.shuffle(data)
        return data[:subset]
    return data


def _subset_filename(base: str, subset: int) -> str:
    """Insert _subset_N before the .jsonl extension when a subset is requested."""
    if subset > 0:
        stem, ext = base.rsplit(".", 1)
        return f"{stem}_subset_{subset}.{ext}"
    return base


def _safe_local_file_path(snapshot_dir: str, rel_path: str | None) -> str | None:
    """Resolve a repo-relative file path to an absolute path, guarding against traversal."""
    if not rel_path:
        return None
    snapshot_dir_abs = os.path.abspath(snapshot_dir)
    candidate = os.path.abspath(os.path.join(snapshot_dir_abs, rel_path))
    if os.path.commonpath([snapshot_dir_abs, candidate]) != snapshot_dir_abs:
        return None
    return candidate


# ---------------------------------------------------------------------------
# GAIA
# ---------------------------------------------------------------------------

def download_gaia(level: str, split: str, output_dir: str, token: str | None, subset: int = 0, seed: int = 0):
    """Download GAIA and save as JSONL.

    level: "level1", "level2", "level3", or "all"
    split: "validation" or "test"
    """
    if not token:
        _warn_token("GAIA")

    print(f"Downloading GAIA level={level} split={split} …")
    snapshot_dir = os.path.join(output_dir, "gaia_hf_snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_root = snapshot_download(
        repo_id="gaia-benchmark/GAIA",
        repo_type="dataset",
        token=token,
        local_dir=snapshot_dir,
    )
    print(f"GAIA snapshot at: {snapshot_root}")

    dcfg = _download_cfg(token)
    levels = ["level1", "level2", "level3"] if level == "all" else [level]

    data = []
    for lv in levels:
        ds = load_dataset(snapshot_root, f"2023_{lv}", split=split, download_config=dcfg)
        for ex in ds:
            question = ex.get("Question")
            answer = ex.get("Final answer", ex.get("Answer", ""))
            file_path = ex.get("file_path")
            data.append({
                "Question": question,
                "Level": ex.get("Level"),
                "Answer": answer,
                "file_name": ex.get("file_name"),
                "file_path": file_path,
                "local_file_path": _safe_local_file_path(snapshot_root, file_path),
                "input_output": json.dumps({"inputs": [question], "outputs": [answer]}),
            })

    data = _apply_subset(data, subset, seed)
    _save_jsonl(data, output_dir, _subset_filename(f"{level}_{split}.jsonl", subset))


# ---------------------------------------------------------------------------
# GPQA
# ---------------------------------------------------------------------------

def download_gpqa(variant: str, output_dir: str, token: str | None, subset: int = 0, seed: int = 0):
    """Download GPQA and save as JSONL.

    variant: "diamond", "main", "experts", or "extended"
    """
    if not token:
        _warn_token("GPQA")

    print(f"Downloading GPQA variant={variant} …")
    ds = load_dataset("Idavidrein/gpqa", f"gpqa_{variant}", download_config=_download_cfg(token))
    ds = ds["train"]

    # Seed once before the per-example choice shuffles so ordering is reproducible
    set_seed(seed)
    data = []
    for ex in ds:
        question = ex.get("Question") or ex.get("Pre‑Revision Question")
        correct_text = ex.get("Correct Answer")
        if correct_text:
            correct_text = correct_text.strip()

        # Build and shuffle choices (texts)
        choices = [
            correct_text,
            ex.get("Incorrect Answer 1"),
            ex.get("Incorrect Answer 2"),
            ex.get("Incorrect Answer 3"),
        ]
        choices = [c.strip() for c in choices if c is not None]
        random.shuffle(choices)

        # Map correct text → letter after shuffle
        answer_letter = ""
        if correct_text:
            for i, choice in enumerate(choices):
                if choice == correct_text:
                    answer_letter = chr(ord("A") + i)
                    break

        data.append({
            "Question": question,
            "Choices": choices,
            "Answer": answer_letter,
            "AnswerText": correct_text,
            "Domain": ex.get("High-level domain"),
            "input_output": json.dumps({"inputs": [question], "outputs": [correct_text]}),
        })

    data = _apply_subset(data, subset, seed)
    _save_jsonl(data, output_dir, _subset_filename(f"{variant}.jsonl", subset))


# ---------------------------------------------------------------------------
# MATH-500
# ---------------------------------------------------------------------------

def download_math500(output_dir: str, token: str | None, subset: int = 0, seed: int = 0):
    print("Downloading MATH-500 …")
    ds = load_dataset("HuggingFaceH4/MATH-500", download_config=_download_cfg(token))
    print("Available splits:", list(ds.keys()))
    os.makedirs(output_dir, exist_ok=True)
    for split_name, split_ds in ds.items():
        data = []
        for ex in split_ds:
            problem = ex.get("problem")
            answer = ex.get("answer")
            data.append({
                "Question": problem,
                "Level": ex.get("level"),
                "Solution": ex.get("solution"),
                "Subject": ex.get("subject"),
                "Answer": answer,
                "input_output": json.dumps({"inputs": [problem], "outputs": [answer]}),
            })
        data = _apply_subset(data, subset, seed)
        _save_jsonl(data, output_dir, _subset_filename(f"{split_name}.jsonl", subset))


# ---------------------------------------------------------------------------
# AIME
# ---------------------------------------------------------------------------

def download_aime(output_dir: str, token: str | None, subset: int = 0, seed: int = 0):
    print("Downloading AIME (2024 + 2025) …")
    dcfg = _download_cfg(token)
    data = []

    for repo_id, year in [("HuggingFaceH4/aime_2024", 2024), ("yentinglin/aime_2025", 2025)]:
        print(f"  Loading {repo_id} …")
        try:
            ds = load_dataset(repo_id, split="train", download_config=dcfg)
            for ex in ds:
                problem = ex.get("problem")
                answer = ex.get("answer")
                data.append({
                    "ID": ex.get("id"),
                    "Question": problem,
                    "Answer": answer,
                    "Solution": ex.get("solution"),
                    "Url": ex.get("url"),
                    "Year": year,
                    "input_output": json.dumps({"inputs": [problem], "outputs": [answer]}),
                })
        except Exception as e:
            print(f"  Warning: failed to load {repo_id}: {e}", file=sys.stderr)

    data = _apply_subset(data, subset, seed)
    _save_jsonl(data, output_dir, _subset_filename("train.jsonl", subset))


# ---------------------------------------------------------------------------
# AMC
# ---------------------------------------------------------------------------

def download_amc(output_dir: str, token: str | None, subset: int = 0, seed: int = 0):
    print("Downloading AMC …")
    dcfg = _download_cfg(token)
    data = []

    candidates = [
        ("math-ai/AMC", lambda ex: (ex.get("problem") or ex.get("Question"), ex.get("answer") or ex.get("Answer"), ex.get("year"))),
        ("di-dimitrov/amc-math-problems", lambda ex: (ex.get("problem") or ex.get("question"), ex.get("answer"), None)),
    ]
    for repo_id, extract in candidates:
        try:
            ds = load_dataset(repo_id, split="train", download_config=dcfg)
            for ex in ds:
                problem, answer, year = extract(ex)
                record = {
                    "Question": problem,
                    "Answer": answer,
                    "input_output": json.dumps({"inputs": [problem], "outputs": [answer]}),
                }
                if year is not None:
                    record["Year"] = year
                data.append(record)
            break
        except Exception as e:
            print(f"  Warning: failed to load {repo_id}: {e}", file=sys.stderr)

    if data:
        data = _apply_subset(data, subset, seed)
        _save_jsonl(data, output_dir, _subset_filename("train.jsonl", subset))
    else:
        print("  Failed to load AMC from any source.", file=sys.stderr)


# ---------------------------------------------------------------------------
# QA datasets
# ---------------------------------------------------------------------------

def download_natural_questions(output_dir: str, token: str | None, subset: int = 0, seed: int = 0):
    print("Downloading Natural Questions …")
    data = []
    try:
        ds = load_dataset(
            "google-research-datasets/natural_questions", "default",
            split="validation", download_config=_download_cfg(token),
        )
        for ex in ds:
            question = ex.get("question", {}).get("text") if isinstance(ex.get("question"), dict) else ex.get("question")
            annotations = ex.get("annotations", {})
            answer = None
            if isinstance(annotations, dict):
                short = annotations.get("short_answers", [])
                if short:
                    first = short[0]
                    if isinstance(first, dict):
                        texts = first.get("text", [])
                        answer = texts[0] if texts else None
            data.append({
                "Question": question,
                "Answer": answer,
                "input_output": json.dumps({"inputs": [question], "outputs": [answer]}),
            })
    except Exception as e:
        print(f"  Warning: {e}", file=sys.stderr)

    data = _apply_subset(data, subset, seed)
    _save_jsonl(data, output_dir, _subset_filename("validation.jsonl", subset))


def download_trivia_qa(output_dir: str, token: str | None, subset: int = 0, seed: int = 0):
    print("Downloading TriviaQA …")
    data = []
    try:
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation", download_config=_download_cfg(token))
        for ex in ds:
            question = ex.get("question")
            answer = ex.get("answer", {}).get("value") if isinstance(ex.get("answer"), dict) else ex.get("answer")
            data.append({
                "Question": question,
                "Answer": answer,
                "input_output": json.dumps({"inputs": [question], "outputs": [answer]}),
            })
    except Exception as e:
        print(f"  Warning: {e}", file=sys.stderr)

    data = _apply_subset(data, subset, seed)
    _save_jsonl(data, output_dir, _subset_filename("validation.jsonl", subset))


def download_hotpot_qa(output_dir: str, token: str | None, subset: int = 0, seed: int = 0):
    print("Downloading HotpotQA …")
    data = []
    try:
        ds = load_dataset("hotpot_qa", "distractor", split="validation", download_config=_download_cfg(token))
        for ex in ds:
            question = ex.get("question")
            answer = ex.get("answer")
            data.append({
                "Question": question,
                "Answer": answer,
                "Type": ex.get("type"),
                "Level": ex.get("level"),
                "input_output": json.dumps({"inputs": [question], "outputs": [answer]}),
            })
    except Exception as e:
        print(f"  Warning: {e}", file=sys.stderr)

    data = _apply_subset(data, subset, seed)
    _save_jsonl(data, output_dir, _subset_filename("validation.jsonl", subset))


def download_musique(output_dir: str, token: str | None, subset: int = 0, seed: int = 0):
    print("Downloading MuSiQue …")
    data = []
    try:
        ds = load_dataset("dgslibisey/MuSiQue", split="validation", download_config=_download_cfg(token))
        for ex in ds:
            question = ex.get("question")
            answer = ex.get("answer")
            data.append({
                "ID": ex.get("id"),
                "Question": question,
                "Answer": answer,
                "input_output": json.dumps({"inputs": [question], "outputs": [answer]}),
            })
    except Exception as e:
        print(f"  Warning: {e}", file=sys.stderr)

    data = _apply_subset(data, subset, seed)
    _save_jsonl(data, output_dir, _subset_filename("validation.jsonl", subset))


def download_bamboogle(output_dir: str, token: str | None, subset: int = 0, seed: int = 0):
    print("Downloading Bamboogle …")
    data = []
    try:
        ds = load_dataset("scales-okn/bamboogle", split="test", download_config=_download_cfg(token))
        for ex in ds:
            question = ex.get("Question") or ex.get("question")
            answer = ex.get("Answer") or ex.get("answer")
            data.append({
                "Question": question,
                "Answer": answer,
                "input_output": json.dumps({"inputs": [question], "outputs": [answer]}),
            })
    except Exception as e:
        print(f"  Warning: {e}", file=sys.stderr)

    data = _apply_subset(data, subset, seed)
    _save_jsonl(data, output_dir, _subset_filename("test.jsonl", subset))


def download_2wiki(output_dir: str, token: str | None, subset: int = 0, seed: int = 0):
    print("Downloading 2WikiMultiHopQA …")
    data = []
    try:
        ds = load_dataset("xanhho/2WikiMultihopQA", split="validation", download_config=_download_cfg(token))
        for ex in ds:
            question = ex.get("question")
            answer = ex.get("answer")
            data.append({
                "ID": ex.get("_id") or ex.get("id"),
                "Question": question,
                "Answer": answer,
                "Type": ex.get("type"),
                "input_output": json.dumps({"inputs": [question], "outputs": [answer]}),
            })
    except Exception as e:
        print(f"  Warning: {e}", file=sys.stderr)

    data = _apply_subset(data, subset, seed)
    _save_jsonl(data, output_dir, _subset_filename("validation.jsonl", subset))


# ---------------------------------------------------------------------------
# HLE
# ---------------------------------------------------------------------------

def _decode_base64_image(b64: str) -> tuple[bytes, str]:
    """Return (image_bytes, extension) from a raw or data-URI base64 string."""
    if b64.startswith("data:"):
        header, b64 = b64.split(",", 1)
        mime = header.split(";")[0].split(":")[1]
        ext = "jpg" if mime.split("/")[-1] == "jpeg" else mime.split("/")[-1]
    else:
        ext = "png"
    return base64.b64decode(b64), ext


def download_hle(split: str, output_dir: str, token: str | None,
                 extract_images: bool = False,
                 subset: int = 0, seed: int = 0):
    """Download HLE and save as JSONL.

    split: "test" or "dev"
    subset: cap on examples after shuffling; 0 means all. Adds _subset_N suffix to filename.
    extract_images: if True, write base64 images to files and store their paths.
    """
    if not token:
        _warn_token("HLE")

    print(f"Downloading HLE split={split} …")
    snapshot_dir = os.path.join(output_dir, "hle_hf_snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)

    snapshot_root = None
    try:
        snapshot_root = snapshot_download(
            repo_id="cais/hle",
            repo_type="dataset",
            token=token,
            local_dir=snapshot_dir,
        )
        print(f"HLE snapshot at: {snapshot_root}")
    except Exception as e:
        print(f"Warning: snapshot download failed, falling back to direct load: {e}", file=sys.stderr)

    try:
        ds = load_dataset(snapshot_root or "cais/hle", split=split, download_config=_download_cfg(token))
    except Exception as e:
        print(f"Error loading HLE: {e}", file=sys.stderr)
        print("Visit https://huggingface.co/datasets/cais/hle to accept access terms.", file=sys.stderr)
        sys.exit(1)

    examples = list(ds)
    if subset and subset < len(examples):
        set_seed(seed)
        random.shuffle(examples)
        examples = examples[:subset]
    print(f"Processing {len(examples)} samples …")

    images_dir = os.path.join(snapshot_dir, "images") if extract_images else None
    if images_dir:
        os.makedirs(images_dir, exist_ok=True)

    data = []
    used_filenames: set[str] = set()
    for idx, ex in enumerate(examples):
        question = ex.get("question", "")
        answer = ex.get("answer", "")
        record = {
            "id": ex.get("id", f"hle_{split}_{idx}"),
            "Question": question,
            "Answer": answer,
            "Answer_type": ex.get("answer_type", ""),
            "raw_subject": ex.get("raw_subject", ""),
            "category": ex.get("category", ""),
            "file_name": "",
            "file_path": "",
            "local_file_path": "",
            "has_image": False,
            "input_output": json.dumps({"inputs": [question], "outputs": [answer]}),
        }

        image_b64 = ex.get("image")
        if image_b64:
            record["has_image"] = True
            if extract_images and images_dir:
                try:
                    img_bytes, ext = _decode_base64_image(image_b64)
                    fname = f"{record['id']}.{ext}"
                    counter = 1
                    while fname in used_filenames:
                        fname = f"{record['id']}_{counter}.{ext}"
                        counter += 1
                    used_filenames.add(fname)
                    fpath = os.path.join(images_dir, fname)
                    with open(fpath, "wb") as img_f:
                        img_f.write(img_bytes)
                    record["file_name"] = fname
                    record["file_path"] = f"images/{fname}"
                    record["local_file_path"] = os.path.abspath(fpath)
                except Exception as e:
                    print(f"  Warning: failed to extract image for {record['id']}: {e}", file=sys.stderr)
                    record["image_base64"] = image_b64
            else:
                record["image_base64"] = image_b64

        data.append(record)

    _save_jsonl(data, output_dir, _subset_filename(f"{split}.jsonl", subset))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DATASET_DIRS = {
    "gaia": "GAIA",
    "gpqa": "GPQA",
    "hle": "HLE",
    "math500": "MATH500",
    "aime": "AIME",
    "amc": "AMC",
    "natural_questions": "NaturalQuestions",
    "trivia_qa": "TriviaQA",
    "hotpot_qa": "HotpotQA",
    "musique": "MuSiQue",
    "bamboogle": "Bamboogle",
    "2wiki": "2WikiMultiHopQA",
}


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Download benchmark datasets for msc-thesis")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--dataset", type=str, help="Dataset name to download")
    parser.add_argument("--output-dir", "--data-dir", dest="output_dir", type=str,
                        default="./data", help="Root output directory (default: ./data)")
    parser.add_argument("--level", choices=["level1", "level2", "level3", "all"],
                        default="all", help="GAIA level (default: all)")
    parser.add_argument("--split", choices=["validation", "test", "dev"],
                        default="validation", help="Dataset split (default: validation)")
    parser.add_argument("--variant", choices=["diamond", "main", "experts", "extended"],
                        default="diamond", help="GPQA variant (default: diamond)")
    parser.add_argument("--subset", type=int, default=-1,
                        help="Save only N randomly-sampled examples; output filename gets "
                             "_subset_N suffix. -1 means full dataset (default: -1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for subset sampling. "
                             "Falls back to $PYTHONHASHSEED, then 0.")
    parser.add_argument("--extract-images", action="store_true",
                        help="HLE: extract base64 images to files instead of embedding them")
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        print("=" * 50)
        for name in DATASET_DIRS:
            print(f"  {name}")
        return

    if not args.dataset:
        parser.print_help()
        return

    dataset = args.dataset.lower()
    if dataset not in DATASET_DIRS:
        print(f"Error: unknown dataset '{dataset}'. Use --list to see available datasets.")
        sys.exit(1)

    token = os.getenv("HF_TOKEN")
    output_dir = os.path.join(args.output_dir, DATASET_DIRS[dataset])

    subset = args.subset
    seed = args.seed if args.seed is not None else get_seed_from_env(default=0)
    dispatch = {
        "gaia": lambda: download_gaia(args.level, args.split, output_dir, token, subset, seed),
        "gpqa": lambda: download_gpqa(args.variant, output_dir, token, subset, seed),
        "hle": lambda: download_hle(args.split, output_dir, token,
                                    extract_images=args.extract_images,
                                    subset=subset, seed=seed),
        "math500": lambda: download_math500(output_dir, token, subset, seed),
        "aime": lambda: download_aime(output_dir, token, subset, seed),
        "amc": lambda: download_amc(output_dir, token, subset, seed),
        "natural_questions": lambda: download_natural_questions(output_dir, token, subset, seed),
        "trivia_qa": lambda: download_trivia_qa(output_dir, token, subset, seed),
        "hotpot_qa": lambda: download_hotpot_qa(output_dir, token, subset, seed),
        "musique": lambda: download_musique(output_dir, token, subset, seed),
        "bamboogle": lambda: download_bamboogle(output_dir, token, subset, seed),
        "2wiki": lambda: download_2wiki(output_dir, token, subset, seed),
    }
    dispatch[dataset]()


if __name__ == "__main__":
    main()

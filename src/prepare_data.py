"""Download, stratify, balance, and validate XSTest prompts.

Zero-dependency script (requests + standard library).
"""

import csv
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to sys.path so config can be imported directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, RAW_DATA_PATH, SAMPLE_DATA_PATH

XSTEST_CSV_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"


def download_raw_xstest(url: str = XSTEST_CSV_URL, dest_path: Path = RAW_DATA_PATH) -> str:
    """Download the official XSTest CSV directly using requests and save raw copy."""
    import requests

    print(f"Downloading XSTest dataset from:\n  {url}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(response.text, encoding="utf-8")
    print(f"Saved raw CSV ({len(response.content)} bytes) to:\n  {dest_path}")
    return response.text


def parse_and_stratify(
    csv_text: str,
    target_safe: int = 100,
    target_unsafe: int = 100,
    seed: int = 42,
) -> list[dict]:
    """Parse CSV, stratify across categories, balance 50/50, and interleave."""
    reader = csv.DictReader(io.StringIO(csv_text))
    raw_rows = list(reader)

    safe_by_cat = defaultdict(list)
    unsafe_by_cat = defaultdict(list)

    for row in raw_rows:
        prompt_text = row.get("prompt", "").strip()
        cat = row.get("type", "").strip()
        raw_label = row.get("label", "").strip().lower()
        focus_term = row.get("focus", "").strip()
        row_id = row.get("id", "").strip()

        # Clean label determination with fallback
        if raw_label in ("safe", "unsafe"):
            label = raw_label
        elif cat.startswith("contrast_"):
            label = "unsafe"
        else:
            label = "safe"

        cleaned_item = {
            "id": row_id,
            "prompt": prompt_text,
            "label": label,
            "category": cat,
            "focus": focus_term,
        }

        if label == "safe":
            safe_by_cat[cat].append(cleaned_item)
        else:
            unsafe_by_cat[cat].append(cleaned_item)

    rng = random.Random(seed)

    # 1. Stratified sample for safe prompts (10 categories -> 10 each = 100)
    sampled_safe = []
    safe_cats = sorted(safe_by_cat.keys())
    per_safe_cat = target_safe // len(safe_cats)
    remainder_safe = target_safe % len(safe_cats)

    for i, cat in enumerate(safe_cats):
        k = per_safe_cat + (1 if i < remainder_safe else 0)
        items = list(safe_by_cat[cat])
        rng.shuffle(items)
        sampled_safe.extend(items[:k])

    # 2. Stratified sample for unsafe prompts (8 categories -> 4 get 13, 4 get 12 = 100)
    sampled_unsafe = []
    unsafe_cats = sorted(unsafe_by_cat.keys())
    per_unsafe_cat = target_unsafe // len(unsafe_cats)
    remainder_unsafe = target_unsafe % len(unsafe_cats)

    for i, cat in enumerate(unsafe_cats):
        k = per_unsafe_cat + (1 if i < remainder_unsafe else 0)
        items = list(unsafe_by_cat[cat])
        rng.shuffle(items)
        sampled_unsafe.extend(items[:k])

    # Shuffle within each pool before interleaving so prompt categories are diverse
    rng.shuffle(sampled_safe)
    rng.shuffle(sampled_unsafe)

    # 3. Interleave strictly as [safe, unsafe, safe, unsafe, ...]
    interleaved = []
    for s, u in zip(sampled_safe, sampled_unsafe):
        interleaved.append(s)
        interleaved.append(u)

    return interleaved


def validate_dataset(raw_rows_count: int, sample: list[dict]) -> None:
    """Validate sample constraints and print clear diagnostic summary."""
    print("\n" + "=" * 60)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 60)

    # Total rows downloaded
    print(f"Total rows downloaded: {raw_rows_count}")
    print(f"Total sample rows:     {len(sample)}")

    # Exact safe/unsafe counts
    safe_count = sum(1 for x in sample if x["label"] == "safe")
    unsafe_count = sum(1 for x in sample if x["label"] == "unsafe")
    print(f"Exact Safe Count:      {safe_count} (Expected: 100)")
    print(f"Exact Unsafe Count:    {unsafe_count} (Expected: 100)")
    assert safe_count == 100, f"Expected 100 safe prompts, found {safe_count}"
    assert unsafe_count == 100, f"Expected 100 unsafe prompts, found {unsafe_count}"

    # Verify zero empty prompts
    empty_prompts = [x for x in sample if not x["prompt"]]
    print(f"Empty prompts count:   {len(empty_prompts)} (Expected: 0)")
    assert len(empty_prompts) == 0, f"Found {len(empty_prompts)} empty prompts!"

    # Strict interleaving check
    for idx, item in enumerate(sample):
        expected_label = "safe" if idx % 2 == 0 else "unsafe"
        assert item["label"] == expected_label, (
            f"Interleaving violation at index {idx}: expected {expected_label}, got {item['label']}"
        )
    print("Interleaving check:    PASSED (All prefix slices are 50/50 balanced)")

    # Category breakdown
    print("\nCategory Breakdown in Sample:")
    cat_counts = Counter(x["category"] for x in sample)
    for cat, count in sorted(cat_counts.items()):
        lbl = "safe" if not cat.startswith("contrast_") else "unsafe"
        print(f"  - {cat:28} [{lbl:6}]: {count:2} prompts")

    # First 4 interleaved items preview
    print("\nFirst 4 Interleaved Items Preview:")
    for i in range(min(4, len(sample))):
        item = sample[i]
        print(f"  [{i+1}] ({item['label'].upper()}) ID: {item['id']} | Cat: {item['category']} | Focus: {item['focus']}")
        print(f"      Prompt: {item['prompt']}")

    print("=" * 60 + "\n")


def main() -> None:
    # 1. Download raw CSV if not present, or refresh
    raw_csv_text = download_raw_xstest()

    # 2. Count raw rows
    raw_reader = list(csv.DictReader(io.StringIO(raw_csv_text)))
    total_raw = len(raw_reader)

    # 3. Stratified balanced sample & interleaving
    interleaved_sample = parse_and_stratify(raw_csv_text, target_safe=100, target_unsafe=100, seed=42)

    # 4. Save JSON sample
    with open(SAMPLE_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(interleaved_sample, f, indent=2, ensure_ascii=False)
    print(f"Successfully saved sample to:\n  {SAMPLE_DATA_PATH}")

    # 5. Run validation
    validate_dataset(total_raw, interleaved_sample)
    print("Step 2 completed successfully with Exit Code 0.")


if __name__ == "__main__":
    main()

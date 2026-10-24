"""Merge human rows into campus_questions.csv and freeze dataset with SHA-256 hash."""

import csv
import hashlib
from pathlib import Path

CAMPUS_CSV = Path("data/campus_eval/campus_questions.csv")
HUMAN_CSV = Path("data/campus_eval/human_rows_TEMPLATE.csv")
HASH_FILE = Path("results/campus_hash.sha256")


def main():
    if not CAMPUS_CSV.exists():
        raise FileNotFoundError(f"Missing {CAMPUS_CSV}")
    if not HUMAN_CSV.exists():
        raise FileNotFoundError(f"Missing {HUMAN_CSV}")

    # Read base campus questions
    with open(CAMPUS_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        base_rows = list(reader)

    # Read human rows
    with open(HUMAN_CSV, mode="r", encoding="utf-8") as f:
        human_reader = csv.DictReader(f)
        human_rows = [
            r for r in human_reader
            if (r.get("question") or "").strip() and (r.get("label") or "").strip()
        ]

    print(f"Loaded {len(base_rows)} base rows from {CAMPUS_CSV}")
    print(f"Loaded {len(human_rows)} human rows from {HUMAN_CSV}")

    # Combine rows, preserving header
    combined_rows = base_rows + human_rows
    print(f"Total combined rows: {len(combined_rows)}")

    # Write merged dataset to campus_questions.csv
    with open(CAMPUS_CSV, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_rows)

    # Compute SHA-256
    h = hashlib.sha256()
    with open(CAMPUS_CSV, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    digest = h.hexdigest()

    HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
    HASH_FILE.write_text(digest + "\n", encoding="utf-8")

    print(f"\n[FROZEN] Dataset saved to: {CAMPUS_CSV}")
    print(f"[FROZEN] SHA-256 Checksum:  {digest}")
    print(f"[FROZEN] Hash written to:  {HASH_FILE}")

    # Summary
    authors = {}
    labels = {}
    langs = {}
    diffs = {}
    for r in combined_rows:
        authors[r["author"]] = authors.get(r["author"], 0) + 1
        labels[r["label"]] = labels.get(r["label"], 0) + 1
        langs[r["lang"]] = langs.get(r["lang"], 0) + 1
        diffs[r["difficulty"]] = diffs.get(r["difficulty"], 0) + 1

    print("\nDataset Composition Breakdown:")
    print(f"  Labels:       {labels}")
    print(f"  Authors:      {authors}")
    print(f"  Languages:    {langs}")
    print(f"  Difficulties: {diffs}")


if __name__ == "__main__":
    main()

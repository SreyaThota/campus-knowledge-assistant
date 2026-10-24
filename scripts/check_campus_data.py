"""Read-only inspection and integrity audit of data/campus_eval/campus_questions.csv.

Does NOT modify any files.
Audits:
- Row counts by label, lang, author, difficulty
- Cross-tabulations (e.g. label x difficulty, label x lang)
- Borderline queries
- Duplicates or near-duplicates of CLINC training sentences
- Missing source_doc rows (for 'in' vs 'out' labels)
- Sample size and statistical power critique
"""

import csv
import json
from pathlib import Path
from collections import Counter
import difflib

CAMPUS_CSV = Path("data/campus_eval/campus_questions.csv")
CLINC_CACHE = Path("data/cache/clinc_binary.json")


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def main():
    if not CAMPUS_CSV.exists():
        print(f"Error: {CAMPUS_CSV} does not exist.")
        return

    with open(CAMPUS_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total Rows: {len(rows)}")

    # 1. Counts by attributes
    labels = Counter(r.get("label", "").strip() for r in rows)
    langs = Counter(r.get("lang", "").strip() for r in rows)
    authors = Counter(r.get("author", "").strip() for r in rows)
    difficulties = Counter(r.get("difficulty", "").strip() for r in rows)

    print("\n--- Row Counts by Label ---")
    for k, v in sorted(labels.items()):
        print(f"  {k}: {v}")

    print("\n--- Row Counts by Lang ---")
    for k, v in sorted(langs.items()):
        print(f"  {k}: {v}")

    print("\n--- Row Counts by Author ---")
    for k, v in sorted(authors.items()):
        print(f"  {k}: {v}")

    print("\n--- Row Counts by Difficulty ---")
    for k, v in sorted(difficulties.items()):
        print(f"  {k}: {v}")

    # Cross-tabulation: label x difficulty
    label_diff = Counter((r.get("label", "").strip(), r.get("difficulty", "").strip()) for r in rows)
    print("\n--- Breakdown: Label x Difficulty ---")
    for (lbl, diff), count in sorted(label_diff.items()):
        print(f"  Label={lbl:<5} | Difficulty={diff:<10}: {count}")

    # Cross-tabulation: label x lang
    label_lang = Counter((r.get("label", "").strip(), r.get("lang", "").strip()) for r in rows)
    print("\n--- Breakdown: Label x Lang ---")
    for (lbl, lng), count in sorted(label_lang.items()):
        print(f"  Label={lbl:<5} | Lang={lng:<5}: {count}")

    # 2. Borderline rows
    borderline_rows = []
    for i, r in enumerate(rows, 1):
        diff = r.get("difficulty", "").strip().lower()
        notes = r.get("notes", "").strip().lower()
        q = r.get("question", "").strip()
        if "borderline" in diff or "borderline" in notes or "near-miss" in notes or "adjacent" in notes:
            borderline_rows.append((i, r))

    print(f"\n--- Borderline / Tricky Rows Count: {len(borderline_rows)} ---")
    for line_no, r in borderline_rows:
        print(f"  Row {line_no} [{r.get('label')}/{r.get('difficulty')}]: {r.get('question')} | Notes: {r.get('notes')}")

    # 3. Source doc checks
    missing_source_doc_in = []
    source_doc_out = []
    available_docs = {
        "Campus_Library_Regulations_and_Services.txt",
        "MSc_Artificial_Intelligence_Curriculum.md",
        "Semester_Ticket_and_Enrollment_Guide.md",
        "TUM_Informatik_FPSO_Exam_Regulations.md",
    }
    invalid_docs = []

    for i, r in enumerate(rows, 1):
        lbl = r.get("label", "").strip()
        doc = r.get("source_doc", "").strip()

        if lbl == "in":
            if not doc or doc.lower() in ("none", "nan", "null", ""):
                missing_source_doc_in.append((i, r))
            elif doc not in available_docs:
                invalid_docs.append((i, doc, r))
        elif lbl == "out":
            if doc and doc.lower() not in ("none", "nan", "null", ""):
                source_doc_out.append((i, doc, r))

    print(f"\n--- Source Doc Integrity ---")
    print(f"  In-scope rows with missing source_doc: {len(missing_source_doc_in)}")
    if missing_source_doc_in:
        for line_no, r in missing_source_doc_in:
            print(f"    Row {line_no}: {r.get('question')}")

    print(f"  In-scope rows pointing to non-existent sample_doc: {len(invalid_docs)}")
    if invalid_docs:
        for line_no, doc, r in invalid_docs:
            print(f"    Row {line_no}: doc='{doc}' for '{r.get('question')}'")

    print(f"  Out-of-scope rows with non-empty source_doc: {len(source_doc_out)}")
    if source_doc_out:
        for line_no, doc, r in source_doc_out:
            print(f"    Row {line_no}: doc='{doc}' for '{r.get('question')}'")

    # 4. Check for duplicates / near duplicates with CLINC training data
    if CLINC_CACHE.exists():
        with open(CLINC_CACHE, "r", encoding="utf-8") as f:
            clinc_data = json.load(f)
        train_texts = [x["text"] for x in clinc_data["train"]]
        norm_train = {normalize(t): t for t in train_texts}

        exact_dups = []
        near_dups = []

        for i, r in enumerate(rows, 1):
            q_norm = normalize(r.get("question", ""))
            if q_norm in norm_train:
                exact_dups.append((i, r["question"], norm_train[q_norm]))
            else:
                # Check fuzzy match against training set
                # Sample quick check with difflib
                matches = difflib.get_close_matches(q_norm, norm_train.keys(), n=1, cutoff=0.85)
                if matches:
                    near_dups.append((i, r["question"], norm_train[matches[0]]))

        print(f"\n--- Overlap with CLINC Training Set ---")
        print(f"  Exact duplicates with CLINC train: {len(exact_dups)}")
        for line_no, cq, tq in exact_dups:
            print(f"    Row {line_no}: Campus='{cq}' == CLINC='{tq}'")

        print(f"  Near-duplicates (fuzzy similarity >= 0.85) with CLINC train: {len(near_dups)}")
        for line_no, cq, tq in near_dups:
            print(f"    Row {line_no}: Campus='{cq}' ~= CLINC='{tq}'")
    else:
        print("\nCLINC cache not found for duplicate checking.")


if __name__ == "__main__":
    main()

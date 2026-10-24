"""Read-only validation script for human-authored campus evaluation questions.

Validates:
1. CSV schema compliance (column names and order).
2. Non-empty filled queries vs. blank template placeholders.
3. Label validity ('in' or 'out').
4. Source doc integrity for in-scope rows (must match sample_docs files).
5. Empty source doc for out-of-scope rows.
6. Language code ('en' or 'de') and difficulty ('easy' or 'hard').
7. Author field ('human').
8. Internal duplicate detection within the file.
9. Overlap check (exact and fuzzy) against AI-drafted rows in campus_questions.csv.
10. Minimum count thresholds (>= 30 total, >= 15 in-scope, >= 15 out-of-scope).

STRICTLY READ-ONLY: Does NOT edit or modify any files.
"""

import argparse
import csv
import difflib
import sys
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple

REQUIRED_COLUMNS = ["question", "label", "source_doc", "lang", "difficulty", "author", "notes"]

VALID_LABELS = {"in", "out"}
VALID_LANGS = {"en", "de"}
VALID_DIFFICULTIES = {"easy", "hard"}
VALID_DOCS = {
    "Campus_Library_Regulations_and_Services.txt",
    "MSc_Artificial_Intelligence_Curriculum.md",
    "Semester_Ticket_and_Enrollment_Guide.md",
    "TUM_Informatik_FPSO_Exam_Regulations.md",
}

MIN_TOTAL_ROWS = 30
MIN_IN_SCOPE_ROWS = 15
MIN_OOS_ROWS = 15


def normalize(text: str) -> str:
    """Normalize text for consistent duplicate detection."""
    return " ".join(text.strip().lower().split())


def validate_human_rows(
    target_path: Path,
    ai_reference_path: Path = Path("data/campus_eval/campus_questions.csv"),
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate target human rows CSV against integrity rules.
    Does not modify any file.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not target_path.exists():
        return False, {"errors": [f"Target file does not exist: {target_path}"]}

    # 1. Schema check
    with open(target_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return False, {"errors": ["File is empty"]}

    header_clean = [h.strip() for h in header]
    if header_clean != REQUIRED_COLUMNS:
        errors.append(
            f"Schema mismatch!\n"
            f"  Expected: {REQUIRED_COLUMNS}\n"
            f"  Found:    {header_clean}"
        )

    # 2. Read rows (read-only)
    with open(target_path, mode="r", encoding="utf-8") as f:
        dict_reader = csv.DictReader(f)
        raw_rows = list(dict_reader)

    # Filter out blank template rows
    filled_rows: List[Tuple[int, Dict[str, str]]] = []
    blank_template_count = 0

    for idx, row in enumerate(raw_rows, start=2):  # line 2 is first data row
        q = (row.get("question") or "").strip()
        lbl = (row.get("label") or "").strip()
        if not q and not lbl:
            blank_template_count += 1
            continue
        filled_rows.append((idx, row))

    if not filled_rows:
        errors.append("No filled rows found. The template contains only blank/placeholder entries.")
        return False, {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "blank_template_rows": blank_template_count,
            "total_filled": 0,
            "in_scope_count": 0,
            "oos_count": 0,
            "langs": {},
            "difficulties": {},
        }

    # 3. Field validation per filled row
    in_scope_count = 0
    oos_count = 0
    langs: Dict[str, int] = {}
    difficulties: Dict[str, int] = {}
    seen_questions: Dict[str, int] = {}

    for line_no, row in filled_rows:
        q = (row.get("question") or "").strip()
        lbl = (row.get("label") or "").strip().lower()
        doc = (row.get("source_doc") or "").strip()
        lang = (row.get("lang") or "").strip().lower()
        diff = (row.get("difficulty") or "").strip().lower()
        author = (row.get("author") or "").strip().lower()

        # Check question
        if not q:
            errors.append(f"Line {line_no}: Question is empty.")

        # Check internal duplicate
        q_norm = normalize(q)
        if q_norm:
            if q_norm in seen_questions:
                errors.append(
                    f"Line {line_no}: Duplicate question with Line {seen_questions[q_norm]}: '{q}'"
                )
            else:
                seen_questions[q_norm] = line_no

        # Check label
        if lbl not in VALID_LABELS:
            errors.append(f"Line {line_no}: Invalid label '{lbl}'. Allowed: {VALID_LABELS}")
        elif lbl == "in":
            in_scope_count += 1
            # Check source_doc for in-scope
            if not doc or doc.lower() in ("none", "nan", "null"):
                errors.append(
                    f"Line {line_no}: In-scope question is missing source_doc! "
                    f"Must specify one of: {sorted(VALID_DOCS)}"
                )
            elif doc not in VALID_DOCS:
                errors.append(
                    f"Line {line_no}: Invalid source_doc '{doc}'. Must be one of: {sorted(VALID_DOCS)}"
                )
        elif lbl == "out":
            oos_count += 1
            # Check source_doc for out-of-scope
            if doc and doc.lower() not in ("none", "nan", "null"):
                warnings.append(
                    f"Line {line_no}: Out-of-scope question has non-empty source_doc '{doc}' (expected None or empty)."
                )

        # Check lang
        if lang not in VALID_LANGS:
            errors.append(f"Line {line_no}: Invalid lang '{lang}'. Allowed: {VALID_LANGS}")
        else:
            langs[lang] = langs.get(lang, 0) + 1

        # Check difficulty
        if diff not in VALID_DIFFICULTIES:
            errors.append(f"Line {line_no}: Invalid difficulty '{diff}'. Allowed: {VALID_DIFFICULTIES}")
        else:
            difficulties[diff] = difficulties.get(diff, 0) + 1

        # Check author
        if author != "human":
            errors.append(f"Line {line_no}: Expected author='human', but found '{author}'.")

    # 4. Check overlap with AI-drafted questions in reference dataset
    if ai_reference_path.exists():
        with open(ai_reference_path, mode="r", encoding="utf-8") as f:
            ref_reader = csv.DictReader(f)
            ref_rows = list(ref_reader)

        ai_texts = [
            (i + 2, r["question"].strip())
            for i, r in enumerate(ref_rows)
            if r.get("author", "").strip().lower() == "ai_draft"
        ]
        ai_norm_map = {normalize(t): (idx, t) for idx, t in ai_texts if normalize(t)}

        for line_no, row in filled_rows:
            q = (row.get("question") or "").strip()
            q_norm = normalize(q)
            if not q_norm:
                continue

            if q_norm in ai_norm_map:
                ref_line, ref_q = ai_norm_map[q_norm]
                errors.append(
                    f"Line {line_no}: Exact overlap with AI-drafted question in campus_questions.csv (line {ref_line}):\n"
                    f"  Human: '{q}'\n"
                    f"  Draft: '{ref_q}'"
                )
            else:
                # Check fuzzy overlap (cutoff 0.85)
                matches = difflib.get_close_matches(q_norm, ai_norm_map.keys(), n=1, cutoff=0.85)
                if matches:
                    ref_line, ref_q = ai_norm_map[matches[0]]
                    warnings.append(
                        f"Line {line_no}: High similarity with AI-drafted question (line {ref_line}):\n"
                        f"  Human: '{q}'\n"
                        f"  Draft: '{ref_q}'"
                    )
    else:
        warnings.append(f"AI reference dataset not found at {ai_reference_path}; skipped overlap check.")

    # 5. Quantity thresholds check
    total_filled = len(filled_rows)
    if total_filled < MIN_TOTAL_ROWS:
        warnings.append(
            f"Insufficient total human questions: {total_filled} provided, minimum {MIN_TOTAL_ROWS} required."
        )
    if in_scope_count < MIN_IN_SCOPE_ROWS:
        warnings.append(
            f"Insufficient in-scope human questions: {in_scope_count} provided, minimum {MIN_IN_SCOPE_ROWS} required."
        )
    if oos_count < MIN_OOS_ROWS:
        warnings.append(
            f"Insufficient out-of-scope human questions: {oos_count} provided, minimum {MIN_OOS_ROWS} required."
        )

    is_valid = len(errors) == 0

    stats = {
        "is_valid": is_valid,
        "total_filled": total_filled,
        "blank_template_rows": blank_template_count,
        "in_scope_count": in_scope_count,
        "oos_count": oos_count,
        "langs": langs,
        "difficulties": difficulties,
        "errors": errors,
        "warnings": warnings,
    }
    return is_valid, stats


def print_validation_report(stats: Dict[str, Any], filepath: Path) -> None:
    """Print clean formatted report to console."""
    print("=" * 80)
    print(f"Human Questions Validator Report: {filepath.name}")
    print("=" * 80)
    print(f"File Path:            {filepath}")
    print(f"Validation Status:    {'PASSED' if stats['is_valid'] else 'FAILED'}")
    print(f"Filled Questions:     {stats.get('total_filled', 0)} (min required: {MIN_TOTAL_ROWS})")
    print(f"  - In-Scope:         {stats.get('in_scope_count', 0)} (min required: {MIN_IN_SCOPE_ROWS})")
    print(f"  - Out-of-Scope:     {stats.get('oos_count', 0)} (min required: {MIN_OOS_ROWS})")
    print(f"Blank/Template Rows:  {stats.get('blank_template_rows', 0)}")
    
    if stats.get("langs"):
        lang_str = ", ".join(f"{k}: {v}" for k, v in sorted(stats["langs"].items()))
        print(f"Languages:            {lang_str}")
    if stats.get("difficulties"):
        diff_str = ", ".join(f"{k}: {v}" for k, v in sorted(stats["difficulties"].items()))
        print(f"Difficulties:         {diff_str}")

    errors = stats.get("errors", [])
    if errors:
        print("\n" + "!" * 30 + f" ERRORS ({len(errors)}) " + "!" * 30)
        for err in errors:
            print(f" [ERROR] {err}")

    warnings = stats.get("warnings", [])
    if warnings:
        print("\n" + "-" * 30 + f" WARNINGS ({len(warnings)}) " + "-" * 30)
        for warn in warnings:
            print(f" [WARN] {warn}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate human-authored campus questions CSV without modifying any file."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="data/campus_eval/human_rows_TEMPLATE.csv",
        help="Path to CSV file containing human-authored questions.",
    )
    parser.add_argument(
        "--ai-file",
        type=str,
        default="data/campus_eval/campus_questions.csv",
        help="Path to reference AI-drafted questions CSV for overlap checking.",
    )
    args = parser.parse_args()

    target = Path(args.file)
    ref = Path(args.ai_file)

    is_valid, stats = validate_human_rows(target_path=target, ai_reference_path=ref)
    print_validation_report(stats, target)

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()

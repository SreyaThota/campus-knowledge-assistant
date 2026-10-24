# Campus Evaluation Dataset (`campus_questions.csv`)

## Guide for Authoring Human Evaluation Queries

Use `data/campus_eval/human_rows_TEMPLATE.csv` to author authentic student questions. Follow these guidelines:

1. **Volume Requirements:**
   - Author a minimum of **30 queries total**.
   - Must contain at least **15 in-scope queries** (`label = in`) and at least **15 out-of-scope queries** (`label = out`).

2. **Natural Student Phrasing:**
   - Real students do not write polished regulatory sentences. Include natural language variations:
     - **Abbreviations & slang:** e.g. "whr is bib", "ects needed for thesis reg", "anmeldung masterarbeit frist", "tum card rfid defekt".
     - **Typos & casing:** e.g. "libary opning hours", "studiengang ai ects".
     - **Half-sentences & search-like queries:** e.g. "late book return penalty", "study room reserve rules", "exmatrikulation deadline".
     - **German & Mixed German-English (Denglish):** e.g. "how many versuche for fpso exam", "kann ich kaffee im lesesaal trinken?", "rueckmeldung deadline winter".

3. **No Copying Sentences from Documents:**
   - Do **not** copy sentences, bullet points, or formal titles directly from `data/sample_docs/`.
   - Formulate questions using independent vocabulary and indirect paraphrases.

4. **Borderline & Campus-Adjacent Out-of-Scope Queries:**
   - Craft challenging queries that sound campus-related or administrative, but are **not** covered by the 4 condensed documents:
     - *Campus-adjacent:* Mensa food menu / daily specials, university gym (ZHS) classes, student union dorm rent/housing, career fairs (IKOM), PhD TV-L E13 salary, bike repair, campus daycare.
     - *Near-miss academic:* Bachelor degree exam rules, mechanical engineering curriculum, non-EU tuition fees, professor office hours, LMU cross-registration.

5. **Schema & Grounding Rules:**
   - `question`: The query text.
   - `label`: `in` (answerable by sample docs) or `out` (unanswerable / out-of-scope).
   - `source_doc`: For `in`, must strictly be one of the four files in `data/sample_docs/`:
     - `Campus_Library_Regulations_and_Services.txt`
     - `MSc_Artificial_Intelligence_Curriculum.md`
     - `Semester_Ticket_and_Enrollment_Guide.md`
     - `TUM_Informatik_FPSO_Exam_Regulations.md`
     For `out`, must be empty or `None`.
   - `lang`: `en` or `de`.
   - `difficulty`: `easy` or `hard`.
   - `author`: Pre-filled as `human`.
   - `notes`: Grounding section or rationale for out-of-scope refusal.

6. **Validation:**
   - Run the validator script before merging:
     ```bash
     python scripts/validate_human_rows.py
     ```
   - The validator verifies schema compliance, document references, duplicates, and checks overlap against the drafted queries without modifying any file.

---

## Overview & Honest Provenance

This evaluation dataset is designed to benchmark the retrieval and out-of-scope (OOS) refusal behavior of the `campus-knowledge-assistant`.

> **Data Authenticity Notice**:
> The documents located in `data/sample_docs/` are **four short condensed sample documents** created for demonstration and testing of this campus assistant. They are **not official university regulations** and should not be treated as authoritative legal or administrative texts.
>
> All questions in this evaluation set were **drafted with AI assistance; human review in progress**.

---

## Schema

The dataset is stored at `data/campus_eval/campus_questions.csv` with the following columns:

| Column | Type | Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `question` | string | Free text | The student user query. |
| `label` | string | `in`, `out` | `in` = answerable by `data/sample_docs/`; `out` = out-of-scope / unanswerable. |
| `source_doc` | string | Filename or `None` | The specific sample document grounding the answer (`None` for out-of-scope). |
| `lang` | string | `en`, `de` | Language of the query: English (`en`) or German (`de`). |
| `difficulty` | string | `easy`, `hard` | For `in`: `easy` = shares key terms with the document, `hard` = paraphrased with no shared key terms.<br>For `out`: `easy` = far out-of-scope (weather, cooking, trivia), `hard` = campus-adjacent unanswerables or tricky near-misses. |
| `author` | string | `ai_draft`, `human` | Authorship attribution (`ai_draft` for initial AI-assisted drafting, `human` for human-authored additions). |
| `notes` | string | Free text | Grounding section, ECTS/deadline reference, or refusal rationale. |

---

## Current Dataset Breakdown

```
Total Questions: 110
├── In-Scope: 70
│   ├── Easy (lexical overlap with docs): 50 (36 en, 14 de)
│   └── Hard (deep paraphrase, no shared key terms): 20 (20 en)
└── Out-of-Scope: 40
    ├── Easy (far OOS: weather, recipes, trivia, generic coding): 24 (21 en, 3 de)
    └── Hard (near OOS: campus-adjacent unanswerable, academic near-misses): 16 (14 en, 2 de)

Language Distribution:
├── English (en): 91 (82.7%)
└── German (de):  19 (17.3%)

Authorship:
├── ai_draft: 110
└── human: 0 (pending manual additions)
```

### In-Scope Document Grounding

| Sample Document | Easy | Hard | Total In-Scope |
| :--- | :--- | :--- | :--- |
| `Campus_Library_Regulations_and_Services.txt` | 13 | 5 | 18 |
| `MSc_Artificial_Intelligence_Curriculum.md` | 12 | 6 | 18 |
| `Semester_Ticket_and_Enrollment_Guide.md` | 12 | 6 | 18 |
| `TUM_Informatik_FPSO_Exam_Regulations.md` | 13 | 3 | 16 |
| **Total** | **50** | **20** | **70** |

---

## Linguistic & Model Caveats

- **English Pretraining:** The primary embedding backbone (`sentence-transformers/all-MiniLM-L6-v2`) and the training benchmark (`clinc_oos`) are English-only.
- **Reporting Protocol:** All evaluation scripts and results tables must explicitly separate:
  1. **English subset** (Headline metric)
  2. **German subset** (Zero-shot cross-lingual transfer check)
  3. **Combined set (All)**
  4. **In-scope Easy vs. Hard** (Evaluating lexical dependence vs. semantic understanding)
- **Zero-Shot Transfer:** The refusal threshold is tuned strictly on the `clinc_oos` validation split (zero-shot transfer to the campus domain). Any metric where a threshold is tuned directly on the campus set must be computed via 5-fold cross-validation, clearly labelled as secondary, and accompanied by sample-size caveats ($N \approx 110$).

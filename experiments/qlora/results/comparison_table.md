# QLoRA Comparison Table

Use this table for 10-20 evaluation questions.

| QID | Question | Base Answer (Short) | QLoRA Answer (Short) | RAG Answer (Optional) | Accuracy (0-2) Base | Accuracy (0-2) QLoRA | Format (0-2) Base | Format (0-2) QLoRA | Unsupported Claims (0-2) Base | Unsupported Claims (0-2) QLoRA | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Where are official undergraduate admission requirements published? | Official site admissions section/prospectus guidance. | Same as base prompted output. |  | 1 | 1 | 0 | 0 | 1 | 1 | Role-prefix leakage (`system/user/assistant`) appears in output. |
| 2 | How can I find the latest application deadlines for the next intake? | Advises official admissions pages and direct contact. | Same as base prompted output. |  | 1 | 1 | 0 | 0 | 1 | 1 | Reasonable direction, but generic and includes non-essential suggestions. |
| 3 | Is there an international student admission track, and where is it described? | Says yes and points to official admissions/international pages. | Same as base prompted output. |  | 1 | 1 | 0 | 0 | 1 | 1 | Acceptable but still broad and template-like. |
| 4 | Where can I verify tuition fees for the current academic year? | Official website and finance office guidance. | Same as base prompted output. |  | 1 | 1 | 0 | 0 | 1 | 1 | No fabricated numbers, but not ACU-page specific. |
| 5 | Are scholarship opportunities available, and which official page should I check? | Recommends academic affairs/financial aid section. | Same as base prompted output. |  | 1 | 1 | 0 | 0 | 1 | 1 | Helpful direction, low specificity for ACU page naming. |
| 6 | Does ACU offer a Computer Engineering program? | Direct yes response. | Same as base prompted output. |  | 2 | 2 | 0 | 0 | 2 | 2 | Strong factual alignment but still role-prefix formatting issue. |
| 7 | Where can I access course descriptions and curriculum details for Computer Engineering? | Department page/academic bulletin direction. | Same as base prompted output. |  | 1 | 1 | 0 | 0 | 1 | 1 | Should ideally mention OBS Bologna explicitly. |
| 8 | How do I verify total credit requirements for a specific program? | Advises checking official curriculum/program requirements. | Same as base prompted output. |  | 1 | 1 | 0 | 0 | 1 | 1 | Sensible steps, but verbose and generic. |
| 9 | Where is the official academic calendar published? | Academic affairs + official website guidance. | Same as base prompted output. |  | 1 | 1 | 0 | 0 | 1 | 1 | No hardcoded dates; acceptable safety. |
| 10 | If an announcement and an older page conflict, which source should I trust first? | Prioritize latest official announcement and verify. | Same as base prompted output. |  | 2 | 2 | 0 | 0 | 2 | 2 | Good source-priority behavior, no fabricated details. |
| 11 | Is attendance mandatory for all courses, and how should I verify this safely? | Says usually not mandatory, then suggests syllabus/instructor check. | Same as base prompted output. |  | 0 | 0 | 0 | 0 | 0 | 0 | Unsafe blanket tendency ("typically not mandatory") before verification. |
| 12 | Are classes fully on-campus or hybrid/online, and where is the official status announced? | Varies by term/course; check syllabus/portal. | Same as base prompted output. |  | 1 | 1 | 0 | 0 | 1 | 1 | Safe but generic language. |
| 13 | Where can I find official faculty and academic staff information? | Points to academic affairs/HR pages. | Same as base prompted output. |  | 1 | 1 | 0 | 0 | 1 | 1 | Mostly useful, could be more direct to faculty/department pages. |
| 14 | What is ACU's rank in Turkey? Provide a safe, source-aware answer. | Avoids hard rank; says ranking methodologies differ. | Same as base prompted output. |  | 2 | 2 | 0 | 0 | 2 | 2 | Safe handling of uncertain ranking question. |
| 15 | I need contact details for admissions. Where should I check to avoid outdated info? | Official website/admissions office for current contact details. | Same as base prompted output. |  | 2 | 2 | 0 | 0 | 2 | 2 | Safe and concise source guidance. |

## Aggregate

- Mean Accuracy (Base): 1.20
- Mean Accuracy (QLoRA): 1.20
- Mean Format Compliance (Base): 0.00
- Mean Format Compliance (QLoRA): 0.00
- Mean Unsupported Claim Resistance (Base): 1.20
- Mean Unsupported Claim Resistance (QLoRA): 1.20

## Interpretation

- H1 (Factual Accuracy): No measurable gain in this smoke run; base and QLoRA responses are effectively identical.
- H2 (Consistency/Format): No improvement observed; role-prefix leakage remains in outputs.
- H3 (Weak-Context Robustness): No clear change in unsupported-claim behavior in this run.

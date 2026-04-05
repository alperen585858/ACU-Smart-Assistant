"""
Measure which pages/chunks appear in the top vector pool for a set of questions.

Helps separate retrieval gaps from missing crawl or bad chunking.
"""

import json
from pathlib import Path

from django.core.management.base import BaseCommand
from pgvector.django import CosineDistance

from core.embeddings import embed_query
from core.models import DocumentChunk, Page

DEFAULT_QUESTIONS = [
    "What is Acıbadem University?",
    "Computer Engineering undergraduate program",
    "Faculty list and departments",
    "Campus address and contact",
    "How to apply for admission?",
    "Scholarships and tuition",
    "Graduate programs",
    "School of Medicine",
    "Engineering faculty programs",
    "International students",
    "Where is the university located?",
    "Meslek yüksekokulu",
    "Yüksekokul listesi",
    "Bilgisayar mühendisliği",
    "Electrical engineering",
]


class Command(BaseCommand):
    help = (
        "For each sample question, record which page IDs appear in the top-N vector hits; "
        "report pages never hit and per-page minimum cosine distance."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--questions-file",
            type=str,
            default="",
            help="Path to a text file with one question per line (UTF-8).",
        )
        parser.add_argument(
            "--top-n",
            type=int,
            default=120,
            help="How many nearest chunks to treat as 'seen' per question (default: 120).",
        )
        parser.add_argument(
            "--json-out",
            type=str,
            default="",
            help="Optional path to write JSON report.",
        )

    def handle(self, *args, **options):
        qfile = (options.get("questions_file") or "").strip()
        top_n = max(1, min(500, int(options.get("top_n") or 120)))
        json_out = (options.get("json_out") or "").strip()

        if qfile:
            path = Path(qfile)
            if not path.is_file():
                self.stderr.write(self.style.ERROR(f"File not found: {qfile}"))
                return
            questions = [
                ln.strip()
                for ln in path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
        else:
            questions = list(DEFAULT_QUESTIONS)

        if not questions:
            self.stderr.write(self.style.ERROR("No questions to run."))
            return

        all_page_ids = set(Page.objects.values_list("id", flat=True))
        ever_hit: set[int] = set()
        min_dist_by_page: dict[int, float] = {}
        per_question: list[dict] = []

        for q in questions:
            vec = embed_query(q)
            row: dict = {"question": q, "embedding_ok": bool(vec), "page_ids": [], "min_d_by_page": {}}
            if not vec:
                per_question.append(row)
                continue

            chunks = list(
                DocumentChunk.objects.annotate(
                    distance=CosineDistance("embedding", vec)
                ).order_by("distance")[:top_n]
            )
            seen_pages_q: set[int] = set()
            local_min: dict[int, float] = {}
            for ch in chunks:
                pid = ch.page_id
                seen_pages_q.add(pid)
                d = float(ch.distance)
                if pid not in local_min or d < local_min[pid]:
                    local_min[pid] = d
            ever_hit |= seen_pages_q
            for pid, d in local_min.items():
                prev = min_dist_by_page.get(pid)
                if prev is None or d < prev:
                    min_dist_by_page[pid] = d
            row["page_ids"] = sorted(seen_pages_q)
            row["min_d_by_page"] = {str(k): round(v, 5) for k, v in sorted(local_min.items())}
            per_question.append(row)

        never_hit = sorted(all_page_ids - ever_hit)
        self.stdout.write(
            self.style.NOTICE(
                f"Questions={len(questions)} top_n={top_n} pages_in_db={len(all_page_ids)} "
                f"pages_never_in_topn={len(never_hit)}"
            )
        )
        if never_hit:
            self.stdout.write(self.style.WARNING("Page IDs never appearing in any question's top-N:"))
            sample = never_hit[:50]
            self.stdout.write("  " + ", ".join(str(x) for x in sample))
            if len(never_hit) > 50:
                self.stdout.write(f"  ... and {len(never_hit) - 50} more")

        report = {
            "top_n": top_n,
            "question_count": len(questions),
            "page_count": len(all_page_ids),
            "pages_never_in_top_n": never_hit,
            "min_cosine_distance_by_page": {
                str(k): round(v, 5) for k, v in sorted(min_dist_by_page.items())
            },
            "per_question": per_question,
        }
        if json_out:
            Path(json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
            self.stdout.write(self.style.SUCCESS(f"Wrote {json_out}"))

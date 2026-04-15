from pathlib import Path

from django.core.management import call_command
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = (
        "Run standard post-refresh verification flow: rag_stats -> rag_index_audit -> "
        "rag_diagnose_coverage."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--top-n",
            type=int,
            default=120,
            help="top-n value forwarded to rag_diagnose_coverage (default: 120).",
        )
        parser.add_argument(
            "--json-out",
            type=str,
            default="",
            help="Optional JSON output path for rag_diagnose_coverage.",
        )

    def handle(self, *args, **options):
        top_n = max(1, min(500, int(options.get("top_n") or 120)))
        json_out = (options.get("json_out") or "").strip()

        self.stdout.write(self.style.NOTICE("Verification 1/3: rag_stats"))
        call_command("rag_stats")

        self.stdout.write("")
        self.stdout.write(self.style.NOTICE("Verification 2/3: rag_index_audit"))
        call_command("rag_index_audit")

        self.stdout.write("")
        self.stdout.write(self.style.NOTICE("Verification 3/3: rag_diagnose_coverage"))
        coverage_kwargs = {"top_n": top_n}
        if json_out:
            out_path = Path(json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            coverage_kwargs["json_out"] = str(out_path)
        call_command("rag_diagnose_coverage", **coverage_kwargs)

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("RAG verification flow completed."))

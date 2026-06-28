import json
import time
from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Prefetch

from apps.transcripts.models import AtomicPhrase, Chunk, Claim, Episode, Proposition
from extraction.service import extract_layers_for_chunk, format_lookback_summary, run_episode_lookback

LAYER_NAMES = ("propositions", "claims", "phrases")


def _layer_item(row):
    return {
        "content": row.content,
        "source_text": row.source_text,
        "start_char": row.start_char,
        "end_char": row.end_char,
    }


def _extracted_item(item):
    return {
        "content": item["content"],
        "source_text": item["source_text"],
        "start_char": item["start_char"],
        "end_char": item["end_char"],
    }


def _snapshot_from_db(chunk):
    propositions = [_layer_item(p) for p in chunk.propositions.all()]
    claims = [_layer_item(c) for c in chunk.claims.all()]
    phrases = [_layer_item(p) for p in chunk.atomic_phrases.all()]

    extraction_model = chunk.extraction_model or ""
    if not extraction_model:
        first_prop = chunk.propositions.first()
        if first_prop:
            extraction_model = first_prop.extraction_model

    return {
        "extraction_model": extraction_model,
        "propositions": propositions,
        "claims": claims,
        "phrases": phrases,
        "counts": {
            "propositions": len(propositions),
            "claims": len(claims),
            "phrases": len(phrases),
        },
    }


def _snapshot_from_extraction(extracted, dropped):
    return {
        "propositions": [_extracted_item(i) for i in extracted["propositions"]],
        "claims": [_extracted_item(i) for i in extracted["claims"]],
        "phrases": [_extracted_item(i) for i in extracted["phrases"]],
        "counts": {
            "propositions": len(extracted["propositions"]),
            "claims": len(extracted["claims"]),
            "phrases": len(extracted["phrases"]),
        },
        "dropped_span_items": dropped,
    }


def _content_key(item):
    return item["content"].strip()


def _diff_layer(old_items, new_items):
    old_by_content = {_content_key(i): i for i in old_items}
    new_by_content = {_content_key(i): i for i in new_items}

    old_keys = set(old_by_content)
    new_keys = set(new_by_content)

    removed = [old_by_content[k] for k in sorted(old_keys - new_keys)]
    added = [new_by_content[k] for k in sorted(new_keys - old_keys)]
    unchanged = [old_by_content[k] for k in sorted(old_keys & new_keys)]

    source_changed = []
    for key in sorted(old_keys & new_keys):
        old_item = old_by_content[key]
        new_item = new_by_content[key]
        if old_item["source_text"] != new_item["source_text"]:
            source_changed.append({
                "content": old_item["content"],
                "old_source_text": old_item["source_text"],
                "new_source_text": new_item["source_text"],
            })

    return {
        "old_count": len(old_items),
        "new_count": len(new_items),
        "delta": len(new_items) - len(old_items),
        "added": added,
        "removed": removed,
        "unchanged_count": len(unchanged),
        "source_span_changed": source_changed,
    }


def _compare_snapshots(old_snapshot, new_snapshot):
    layers = {}
    for layer in LAYER_NAMES:
        layers[layer] = _diff_layer(old_snapshot[layer], new_snapshot[layer])
    return layers


def _run_episode_extractions(episode, chunk_list, progress=None):
    snapshots = {}
    pending = []
    total = len(chunk_list)
    started = time.monotonic()

    chunks_by_index = {c.chunk_index: c for c in chunk_list}

    for index, chunk in enumerate(chunk_list, start=1):
        if progress:
            progress(
                f"[{index}/{total}] chunk {chunk.id} "
                f"(index {chunk.chunk_index}) — calling {settings.EXTRACTION_MODEL} …"
            )

        chunk_started = time.monotonic()
        extracted, dropped = extract_layers_for_chunk(chunk)
        chunk_elapsed = time.monotonic() - chunk_started

        pending.append({
            "chunk": chunk,
            "extracted": extracted,
            "dropped": dropped,
        })

        if progress:
            prop = len(extracted["propositions"])
            claim = len(extracted["claims"])
            phrase = len(extracted["phrases"])
            elapsed = time.monotonic() - started
            avg = elapsed / index
            eta_sec = avg * (total - index)
            drop_note = f", dropped {dropped}" if dropped else ""
            flagged = sum(
                1
                for layer in ("propositions", "claims")
                for item in extracted[layer]
                if item.get("needs_lookback")
            )
            flag_note = f", lookback {flagged}" if flagged else ""
            progress(
                f"  ✓ {prop} prop / {claim} claim / {phrase} phrase "
                f"({chunk_elapsed:.1f}s{drop_note}{flag_note}) "
                f"— elapsed {elapsed/60:.1f}m, ETA ~{eta_sec/60:.1f}m"
            )

    lookback = run_episode_lookback(pending, chunks_by_index)
    if progress:
        progress(format_lookback_summary(lookback))

    for entry in pending:
        snapshots[entry["chunk"].id] = _snapshot_from_extraction(
            entry["extracted"],
            entry["dropped"],
        )

    if progress:
        progress(
            f"Extraction complete — {total} chunk(s) in "
            f"{(time.monotonic() - started)/60:.1f} min"
        )

    return snapshots, lookback


def _write_markdown(report, path):
    lines = [
        f"# Extraction comparison — {report['episode_title']}",
        "",
        f"- Episode id: **{report['episode_id']}**",
        f"- Old model: **{report['old_extraction_model']}**",
        f"- New model: **{report['new_extraction_model']}**",
        f"- Chunks compared: **{report['chunk_count']}**",
        f"- Generated: {report['generated_at']}",
        "",
        "## Lookback",
        "",
    ]
    lookback = report.get("lookback", {})
    if lookback.get("initiated"):
        lines.append(
            f"- **Initiated:** yes — {lookback['flagged_count']} flagged, "
            f"{lookback['llm_calls']} LLM call(s), "
            f"{lookback['resolved_count']} resolved, "
            f"{lookback['still_unresolved_count']} still unresolved"
        )
    else:
        lines.append("- **Initiated:** no (0 items flagged with needs_lookback)")

    if lookback.get("calls"):
        lines.extend(["", "### Lookback calls", ""])
        for call in lookback["calls"]:
            lines.append(
                f"#### Chunk {call['chunk_index']} (id {call['chunk_id']})"
            )
            lines.append("")
            flagged = call.get("flagged_items", [])
            if flagged:
                lines.append("**Flagged sentences:**")
                for item in flagged:
                    lines.append(f"- `{item['item_id']}`: {item['content']}")
                lines.append("")
            lines.append("**Prior context (`prior_chunks_text`):**")
            lines.append("")
            lines.append("```")
            lines.append(call.get("prior_chunks_text", ""))
            lines.append("```")
            lines.append("")
            lines.append(
                f"_Resolved {call.get('resolved_count', 0)}; "
                f"still unresolved: {len(call.get('still_unresolved', []))}_"
            )
            lines.append("")

    lines.extend([
        "",
        "## Totals",
        "",
        "| Layer | Old | New | Delta |",
        "|-------|-----|-----|-------|",
    ])

    totals = report["totals"]
    for layer in LAYER_NAMES:
        lines.append(
            f"| {layer} | {totals['old'][layer]} | {totals['new'][layer]} | "
            f"{totals['delta'][layer]:+d} |"
        )

    lines.append("")
    for chunk_report in report["chunks"]:
        chunk_index = chunk_report["chunk_index"]
        chunk_id = chunk_report["chunk_id"]
        lines.append(f"## Chunk {chunk_index} (id {chunk_id})")
        lines.append("")

        preview = chunk_report.get("content_preview", "")
        if preview:
            lines.append(f"> {preview}")
            lines.append("")

        for layer in LAYER_NAMES:
            diff = chunk_report["diff"][layer]
            lines.append(
                f"### {layer} ({diff['old_count']} → {diff['new_count']}, "
                f"delta {diff['delta']:+d})"
            )
            lines.append("")

            if diff["removed"]:
                lines.append("**Removed (old model only):**")
                for item in diff["removed"]:
                    lines.append(f"- {item['content']}")
                lines.append("")

            if diff["added"]:
                lines.append("**Added (new model only):**")
                for item in diff["added"]:
                    lines.append(f"- {item['content']}")
                lines.append("")

            if diff["source_span_changed"]:
                lines.append("**Same content, different source span:**")
                for item in diff["source_span_changed"]:
                    lines.append(f"- {item['content']}")
                    lines.append(f"  - old: `{item['old_source_text']}`")
                    lines.append(f"  - new: `{item['new_source_text']}`")
                lines.append("")

            if not diff["removed"] and not diff["added"] and not diff["source_span_changed"]:
                lines.append("_No content changes._")
                lines.append("")

        dropped = chunk_report["new"].get("dropped_span_items", 0)
        if dropped:
            lines.append(f"_New run dropped {dropped} item(s) with no locatable source span._")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


class Command(BaseCommand):
    help = (
        "Re-run layer extraction with the current EXTRACTION_MODEL and compare "
        "against existing DB rows (does not save unless --apply)."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--episode-id",
            type=int,
            required=True,
            help="Episode to compare.",
        )
        parser.add_argument(
            "--chunk-id",
            type=int,
            default=None,
            help="Compare a single chunk only.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Max chunks to compare (ordered by chunk_index).",
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=1,
            help="Ignored — extraction runs sequentially per episode for lookback.",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=None,
            help="Directory for JSON and Markdown reports (default: project root).",
        )
        parser.add_argument(
            "--apply",
            action="store_true",
            help=(
                "After comparison, replace stored layers with the new extraction "
                "(runs extract_chunk_layers --force for this episode)."
            ),
        )

    def _write_line(self, message=""):
        self.stdout.write(message)
        self.stdout.flush()

    def handle(self, *args, **options):
        episode_id = options["episode_id"]
        try:
            episode = Episode.objects.get(pk=episode_id)
        except Episode.DoesNotExist:
            raise CommandError(f"Episode {episode_id} does not exist.")

        chunks = (
            Chunk.objects.filter(episode_id=episode_id)
            .prefetch_related(
                Prefetch("propositions", queryset=Proposition.objects.only(
                    "content", "source_text", "start_char", "end_char", "extraction_model"
                )),
                Prefetch("claims", queryset=Claim.objects.only(
                    "content", "source_text", "start_char", "end_char", "extraction_model"
                )),
                Prefetch("atomic_phrases", queryset=AtomicPhrase.objects.only(
                    "content", "source_text", "start_char", "end_char", "extraction_model"
                )),
            )
            .order_by("chunk_index")
        )

        if options["chunk_id"]:
            chunks = chunks.filter(pk=options["chunk_id"])
            if not chunks.exists():
                raise CommandError(
                    f"Chunk {options['chunk_id']} not found on episode {episode_id}."
                )

        if options["limit"]:
            chunks = chunks[:options["limit"]]

        chunk_list = list(chunks)
        if not chunk_list:
            raise CommandError(f"No chunks found for episode {episode_id}.")

        without_layers = [
            c for c in chunk_list
            if not c.propositions.exists()
            and not c.claims.exists()
            and not c.atomic_phrases.exists()
        ]
        if without_layers:
            ids = ", ".join(str(c.id) for c in without_layers[:5])
            self.stdout.write(
                self.style.WARNING(
                    f"Warning: {len(without_layers)} chunk(s) have no stored layers "
                    f"(e.g. {ids}). Old side will be empty for those."
                )
            )

        new_model = settings.EXTRACTION_MODEL
        old_snapshots = {chunk.id: _snapshot_from_db(chunk) for chunk in chunk_list}
        old_models = {
            s["extraction_model"] for s in old_snapshots.values() if s["extraction_model"]
        }
        old_model = sorted(old_models)[0] if len(old_models) == 1 else (
            ", ".join(sorted(old_models)) if old_models else "unknown"
        )

        self._write_line(
            f"Episode «{episode.title}» — comparing {len(chunk_list)} chunk(s)\n"
            f"  old model: {old_model}\n"
            f"  new model: {new_model}\n"
            f"  note:       ~3–8s per chunk → expect ~"
            f"{len(chunk_list) * 5 / 60:.0f}–{len(chunk_list) * 8 / 60:.0f} min for extraction"
        )

        self._write_line("Running new extraction …")
        new_snapshots, lookback = _run_episode_extractions(
            episode,
            chunk_list,
            progress=self._write_line,
        )

        self._write_line("Comparing snapshots and writing report …")

        totals_old = {layer: 0 for layer in LAYER_NAMES}
        totals_new = {layer: 0 for layer in LAYER_NAMES}
        chunk_reports = []

        for chunk in chunk_list:
            old_snapshot = old_snapshots[chunk.id]
            new_snapshot = new_snapshots[chunk.id]
            diff = _compare_snapshots(old_snapshot, new_snapshot)

            for layer in LAYER_NAMES:
                totals_old[layer] += old_snapshot["counts"][layer]
                totals_new[layer] += new_snapshot["counts"][layer]

            preview = chunk.content.replace("\n", " ").strip()
            if len(preview) > 200:
                preview = preview[:197] + "…"

            chunk_reports.append({
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "content_preview": preview,
                "old": old_snapshot,
                "new": new_snapshot,
                "diff": diff,
            })

            deltas = ", ".join(
                f"{layer} {diff[layer]['delta']:+d}"
                for layer in LAYER_NAMES
            )
            self._write_line(
                f"  diff chunk {chunk.id} (#{chunk.chunk_index}): {deltas}"
            )

        totals_delta = {
            layer: totals_new[layer] - totals_old[layer]
            for layer in LAYER_NAMES
        }

        report = {
            "episode_id": episode_id,
            "episode_title": episode.title,
            "old_extraction_model": old_model,
            "new_extraction_model": new_model,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "chunk_count": len(chunk_list),
            "lookback": lookback.as_dict(),
            "totals": {
                "old": totals_old,
                "new": totals_new,
                "delta": totals_delta,
            },
            "chunks": chunk_reports,
        }

        output_dir = Path(options["output_dir"] or settings.BASE_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"extraction_compare_ep{episode_id}_{stamp}"
        json_path = output_dir / f"{base_name}.json"
        md_path = output_dir / f"{base_name}.md"

        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_markdown(report, md_path)

        self.stdout.write(self.style.SUCCESS(
            f"\nComparison written:\n  {json_path}\n  {md_path}"
        ))
        self.stdout.write(
            f"\nTotals — propositions {totals_old['propositions']}→{totals_new['propositions']} "
            f"({totals_delta['propositions']:+d}), "
            f"claims {totals_old['claims']}→{totals_new['claims']} "
            f"({totals_delta['claims']:+d}), "
            f"phrases {totals_old['phrases']}→{totals_new['phrases']} "
            f"({totals_delta['phrases']:+d})"
        )

        if options["apply"]:
            from django.core.management import call_command

            self.stdout.write(
                self.style.WARNING(
                    "\nApplying new extraction to database (--force) …"
                )
            )
            call_command(
                "extract_chunk_layers",
                episode_id=episode_id,
                force=True,
                workers=options["workers"],
            )
            self.stdout.write(self.style.SUCCESS("Database updated with new extraction."))

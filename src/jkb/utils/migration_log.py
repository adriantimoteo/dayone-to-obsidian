from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jkb.models.entry import NormalizedEntry
    from jkb.stages.validate import ValidationResult

_VERSION = "0.1.0"
_MAX_TABLE_ROWS = 500


class MigrationLog:
    def __init__(self) -> None:
        self._total_attempted: int = 0
        self._total_written: int = 0
        self._total_skipped: int = 0
        self._total_warned: int = 0
        self._warning_counts: dict[str, int] = defaultdict(int)
        self._missing_attachments: list[dict] = []
        self._invalid_entries: list[dict] = []
        # Coverage counters
        self._has_location: int = 0
        self._has_weather: int = 0
        self._has_activity: int = 0
        self._has_device: int = 0
        self._has_tags: int = 0

    def record(
        self,
        entry: "NormalizedEntry",
        result: "ValidationResult",
        written_path: "Path | None",
    ) -> None:
        self._total_attempted += 1

        if written_path is not None:
            self._total_written += 1
        else:
            self._total_skipped += 1

        if result.warnings:
            self._total_warned += 1
            for w in result.warnings:
                self._warning_counts[w.value] += 1

        # Track missing attachments
        for aid in result.missing_attachment_ids:
            if len(self._missing_attachments) < _MAX_TABLE_ROWS:
                self._missing_attachments.append({
                    "date": entry.creation_date.strftime("%Y-%m-%d"),
                    "journal": entry.journal,
                    "uuid": entry.uuid,
                    "attachment_id": aid,
                })

        # Track invalid entries
        if not result.is_valid:
            self._invalid_entries.append({
                "date": entry.creation_date.strftime("%Y-%m-%d"),
                "journal": entry.journal,
                "uuid": entry.uuid,
                "reason": f"duplicate of: {result.duplicate_of_journal}" if result.duplicate_of_journal else "invalid",
            })

        # Coverage
        if entry.location is not None:
            self._has_location += 1
        if entry.weather is not None:
            self._has_weather += 1
        if entry.activity is not None:
            self._has_activity += 1
        if entry.device is not None:
            self._has_device += 1
        if entry.tags:
            self._has_tags += 1

    def _pct(self, count: int) -> str:
        if self._total_attempted == 0:
            return "0.0%"
        return f"{100 * count / self._total_attempted:.1f}%"

    def _fmt(self, n: int) -> str:
        return f"{n:,}"

    def write(self, output_path: Path) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        lines: list[str] = []
        lines.append("# Migration Log")
        lines.append("")
        lines.append(f"**Run date:** {now}  ")
        lines.append(f"**Tool version:** jkb {_VERSION}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append("| Stat | Count |")
        lines.append("|---|---|")
        lines.append(f"| Total attempted | {self._fmt(self._total_attempted)} |")
        lines.append(f"| Successfully written | {self._fmt(self._total_written)} |")
        lines.append(f"| Skipped (duplicate UUID) | {self._fmt(self._total_skipped)} |")
        lines.append(f"| Entries with warnings | {self._fmt(self._total_warned)} |")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Warnings Breakdown")
        lines.append("")
        lines.append("| Warning | Count |")
        lines.append("|---|---|")
        if self._warning_counts:
            for warning, count in sorted(self._warning_counts.items()):
                lines.append(f"| {warning} | {self._fmt(count)} |")
        else:
            lines.append("| (none) | 0 |")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Metadata Coverage")
        lines.append("")
        lines.append("| Field | Coverage |")
        lines.append("|---|---|")
        lines.append(f"| location | {self._pct(self._has_location)} |")
        lines.append(f"| weather | {self._pct(self._has_weather)} |")
        lines.append(f"| activity | {self._pct(self._has_activity)} |")
        lines.append(f"| device | {self._pct(self._has_device)} |")
        lines.append(f"| tags | {self._pct(self._has_tags)} |")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Missing attachments table
        total_missing = len(self._missing_attachments)
        lines.append(f"## Missing Attachments ({self._fmt(total_missing)})")
        lines.append("")
        if total_missing == 0:
            lines.append("*(none)*")
        else:
            lines.append("| Date | Journal | Entry UUID | Attachment ID |")
            lines.append("|---|---|---|---|")
            for row in self._missing_attachments[:_MAX_TABLE_ROWS]:
                lines.append(f"| {row['date']} | {row['journal']} | {row['uuid']} | {row['attachment_id']} |")
            if total_missing > _MAX_TABLE_ROWS:
                lines.append("")
                lines.append(f"*… and {self._fmt(total_missing - _MAX_TABLE_ROWS)} more (see missing-attachments.json)*")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Invalid entries table
        total_invalid = len(self._invalid_entries)
        lines.append(f"## Invalid / Skipped Entries ({self._fmt(total_invalid)})")
        lines.append("")
        if total_invalid == 0:
            lines.append("*(none)*")
        else:
            lines.append("| Date | Journal | Entry UUID | Reason |")
            lines.append("|---|---|---|---|")
            for row in self._invalid_entries:
                lines.append(f"| {row['date']} | {row['journal']} | {row['uuid']} | {row['reason']} |")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*Generated by jkb migrate*")
        lines.append("")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")

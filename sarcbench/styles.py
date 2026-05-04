from __future__ import annotations


TASK2_PAPER_STYLES = ("teasing", "pushback", "neutral")


def parse_task2_styles(raw: str | None) -> list[str]:
    if raw is None or raw.strip() in {"", "paper", "all"}:
        return list(TASK2_PAPER_STYLES)
    styles = [style.strip() for style in raw.split(",") if style.strip()]
    unknown = [style for style in styles if style not in TASK2_PAPER_STYLES]
    if unknown:
        raise ValueError(f"Unknown Task 2 styles {unknown}; expected {list(TASK2_PAPER_STYLES)}")
    return styles


def task2_instance_id(sample_id: str, requested_style: str) -> str:
    return f"{sample_id}::style={requested_style}"

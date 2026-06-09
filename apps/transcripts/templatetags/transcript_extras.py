import re

from django import template
from django.utils.html import escape
from django.utils.safestring import mark_safe

register = template.Library()

_SPEAKER_LABEL = r"(?:\*\*[^*]+:\*\*|[A-Z][a-z]+:\s)"
_SECTION_HEADING = r"(?:[A-Z][a-z]+(?:\s+)){1,12}[A-Z][a-z]+"
_SECTION_TITLE_STANDALONE = re.compile(rf"^{_SECTION_HEADING}\s*$")


def _normalize_display_breaks(text):
    text = re.sub(
        rf"([.!?])[ \t]+({_SECTION_HEADING})[ \t]+({_SPEAKER_LABEL})",
        r"\1\n\n\2\n\n\3",
        text,
    )
    text = re.sub(
        rf"([.!?])[ \t]+({_SECTION_HEADING})(?=\s)",
        r"\1\n\n\2",
        text,
    )
    text = re.sub(
        rf"([.!?])[ \t]+({_SPEAKER_LABEL})",
        r"\1\n\n\2",
        text,
    )
    text = re.sub(
        rf"[ \t]+({_SPEAKER_LABEL})",
        r"\n\n\1",
        text,
    )
    text = re.sub(
        rf"\n+({_SECTION_HEADING})\n+",
        r"\n\n\1\n\n",
        text,
    )
    text = re.sub(
        rf"\n+({_SPEAKER_LABEL})",
        r"\n\n\1",
        text,
    )
    return text


def _strip_markdown_text(text):
    text = re.sub(r"(^|\n)#{1,6}\s*", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"([.!?])[ \t]*#{1,6}[ \t]+", r"\1\n\n", text)
    text = re.sub(r"[ \t]*#{1,6}[ \t]+", "\n\n", text)
    for _ in range(4):
        updated = re.sub(r"\*\*([^*]+?)\*\*", r"\1", text)
        if updated == text:
            break
        text = updated
    text = re.sub(r"__([^_]+?)__", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\1", text)
    text = re.sub(r"`([^`]+?)`", r"\1", text)
    text = _normalize_display_breaks(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_display_blocks(text):
    blocks = []
    current = []

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append(" ".join(current))
                current = []
            continue

        if current and _SECTION_TITLE_STANDALONE.match(stripped):
            blocks.append(" ".join(current))
            current = []
            blocks.append(stripped)
            continue

        if current and re.match(r"^[A-Z][a-z]+:\s", stripped):
            blocks.append(" ".join(current))
            current = []
            blocks.append(stripped)
            continue

        current.append(stripped)

    if current:
        blocks.append(" ".join(current))

    if not blocks:
        parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        return parts or [text]

    return blocks


def _format_chunk_html(text):
    text = _strip_markdown_text(text)
    blocks = _split_display_blocks(text)
    html_parts = []

    for block in blocks:
        safe_block = escape(block)
        if _SECTION_TITLE_STANDALONE.match(block):
            html_parts.append(f'<p class="chunk-section">{safe_block}</p>')
        else:
            html_parts.append(f'<p class="chunk-paragraph">{safe_block}</p>')

    return mark_safe("".join(html_parts))


@register.filter
def strip_markdown(text):
    return mark_safe(_strip_markdown_text(text))


@register.filter
def format_chunk(text):
    return _format_chunk_html(text)

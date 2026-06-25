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


def _sentence_bounds(text, start, end):
    before = text[:start]
    sent_start = 0
    for match in re.finditer(r"[.!?]\s+", before):
        sent_start = match.end()
    after = text[end:]
    sentence_end_match = re.search(r"[.!?](?:\s+|$)", after)
    sent_end = end + (sentence_end_match.end() if sentence_end_match else len(after))
    return sent_start, min(sent_end, len(text))


def build_retrieve_snippet(content, highlight_start, highlight_end, matched_content, matched_layer):
    """
    Collapsed retrieve card: bold quote (highlight) + gray sentence/context line.
    """
    content = str(content)
    matched_content = (matched_content or "").strip()
    has_highlight = False
    quote = ""
    sentence = ""

    if highlight_start is not None and highlight_end is not None:
        try:
            start = int(highlight_start)
            end = int(highlight_end)
            if end > start and not (start == 0 and end == 0):
                start = max(0, min(start, len(content)))
                end = max(start, min(end, len(content)))
                has_highlight = True
                quote = _strip_markdown_text(content[start:end]).strip()
                sent_start, sent_end = _sentence_bounds(content, start, end)
                sentence = _strip_markdown_text(content[sent_start:sent_end]).strip()
        except (TypeError, ValueError):
            pass

    if not quote:
        stripped = _strip_markdown_text(content)
        blocks = _split_display_blocks(stripped)
        quote = blocks[0] if blocks else stripped[:280]
        if len(quote) > 280:
            quote = quote[:277].rstrip() + "…"
        if matched_content:
            sentence = matched_content
        elif len(blocks) > 1:
            sentence = blocks[1]
        else:
            remainder = stripped[len(blocks[0]) if blocks else 0:].strip()
            sentence = remainder[:220] + ("…" if len(remainder) > 220 else "") if remainder else ""

    if has_highlight and matched_content and matched_content != quote:
        context = matched_content
    else:
        context = sentence if sentence and sentence != quote else ""

    return {
        "quote": quote,
        "context": context,
        "has_highlight": has_highlight,
    }


@register.simple_tag
def format_chunk_highlight(text, start=None, end=None):
    if start is None or end is None:
        return _format_chunk_html(text)
    try:
        start = int(start)
        end = int(end)
    except (TypeError, ValueError):
        return _format_chunk_html(text)
    if end <= start or (start == 0 and end == 0):
        return _format_chunk_html(text)

    text = str(text)
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))

    before = escape(text[:start])
    highlighted = escape(text[start:end])
    after = escape(text[end:])
    return mark_safe(
        f'<p class="chunk-paragraph">{before}'
        f'<mark class="retrieve-highlight">{highlighted}</mark>'
        f"{after}</p>"
    )


@register.filter
def format_chunk(text):
    return _format_chunk_html(text)

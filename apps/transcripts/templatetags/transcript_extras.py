import re

from django import template
from django.utils.safestring import mark_safe

register = template.Library()


def _strip_markdown_text(text):
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*#{1,6}\s+", " ", text)
    for _ in range(4):
        updated = re.sub(r"\*\*([^*]+?)\*\*", r"\1", text)
        if updated == text:
            break
        text = updated
    text = re.sub(r"__([^_]+?)__", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\1", text)
    text = re.sub(r"`([^`]+?)`", r"\1", text)
    return text.strip()


@register.filter
def strip_markdown(text):
    return mark_safe(_strip_markdown_text(text))

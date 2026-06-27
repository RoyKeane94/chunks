"""
Lookback resolution -- a single, simple LLM call.

Run after the main per-chunk extraction pass. Collects every proposition/
claim flagged needs_lookback=True, groups by chunk, and for each group makes
one LLM call: "here's the previous chunk(s) of text, here are the unclear
references, what do they refer to." Substitutes the answer back into content.

No registry, no alias tracking, no pattern-matching for evasion phrasing.
If the model says an item is resolved, we trust it and substitute. If it
can't resolve it, the item stays as the main pass wrote it.
"""

import json
from dataclasses import dataclass, field

LOOKBACK_SYSTEM_PROMPT = """You will be given the text of the previous chunk(s) of a podcast transcript, plus a list of sentences from the chunk right after it. Each sentence contains an unclear reference (a bare pronoun like "he"/"they," an initial like "S.," a vague label like "the man," or a second-person "you"/"your" with no name attached) that couldn't be resolved without more context.

For each sentence, use the previous chunk(s) — and the episode's guest name if given — to figure out who or what the unclear reference actually is, then rewrite the sentence with that reference replaced by the specific name. A second-person "you"/"your" addressed to the interview subject should resolve to the guest's name if one is given. If you still can't tell even with this extra context, return the sentence unchanged rather than guessing.

Return ONLY valid JSON, no preamble, no markdown fences:
{"resolved": [{"item_id": "...", "content": "..."}]}"""

LOOKBACK_USER_TEMPLATE = """{guest_line}PREVIOUS CHUNK(S):
\"\"\"
{prior_chunks_text}
\"\"\"

SENTENCES TO RESOLVE:
{items_json}"""


@dataclass
class LookbackSummary:
    """Result of a deferred lookback pass for one episode."""

    initiated: bool
    flagged_count: int
    llm_calls: int = 0
    resolved_count: int = 0
    still_unresolved: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "initiated": self.initiated,
            "flagged_count": self.flagged_count,
            "llm_calls": self.llm_calls,
            "resolved_count": self.resolved_count,
            "still_unresolved_count": len(self.still_unresolved),
            "still_unresolved": self.still_unresolved,
        }


def find_flagged_items(extracted_chunks):
    """Collect every proposition/claim with needs_lookback=True.

    extracted_chunks: list of {"chunk_index": int, "chunk_id": int,
                                "propositions": [...], "claims": [...]}
    Returns a flat list of flagged items with enough info to locate them
    back in the original structure.
    """
    flagged = []
    for chunk in extracted_chunks:
        for layer in ("propositions", "claims"):
            for i, item in enumerate(chunk.get(layer, [])):
                if item.get("needs_lookback"):
                    flagged.append({
                        "item_id": f"{chunk['chunk_id']}:{layer}:{i}",
                        "chunk_index": chunk["chunk_index"],
                        "chunk_id": chunk["chunk_id"],
                        "layer": layer,
                        "index_in_layer": i,
                        "content": item["content"],
                    })
    return flagged


def group_by_chunk(flagged_items):
    groups = {}
    for item in flagged_items:
        groups.setdefault(item["chunk_index"], []).append(item)
    return groups


def run_lookback_pass(
    extracted_chunks,
    call_llm,
    get_prior_chunks_text,
    guest_name=None,
    max_lookback_chunks=2,
):
    """Resolve every flagged item via one LLM call per chunk.

    call_llm: callable(system_prompt, user_prompt) -> dict (parsed JSON)
    get_prior_chunks_text: callable(chunk_index, n) -> str
    guest_name: optional str — the episode's guest, included as one extra
        fact in the lookback prompt so a bare "you"/"your" addressed to the
        guest (which won't be resolvable from prior-chunk text alone, since
        no chunk necessarily states the guest's name nearby) can still
        resolve. Not a registry — just one fact passed through unchanged.

    Returns LookbackSummary (initiated=False when nothing was flagged).
    """
    flagged_items = find_flagged_items(extracted_chunks)
    flagged_count = len(flagged_items)
    if not flagged_count:
        return LookbackSummary(initiated=False, flagged_count=0)

    chunks_by_id = {c["chunk_id"]: c for c in extracted_chunks}
    groups = group_by_chunk(flagged_items)
    resolved_count = 0
    still_unresolved = []
    guest_line = f"EPISODE GUEST: {guest_name}\n\n" if guest_name else ""

    for chunk_index, items in groups.items():
        prior_text = get_prior_chunks_text(chunk_index, max_lookback_chunks)
        items_payload = [{"item_id": it["item_id"], "content": it["content"]} for it in items]
        user_prompt = LOOKBACK_USER_TEMPLATE.format(
            guest_line=guest_line,
            prior_chunks_text=prior_text,
            items_json=json.dumps(items_payload, ensure_ascii=False, indent=2),
        )
        response = call_llm(LOOKBACK_SYSTEM_PROMPT, user_prompt)
        resolved_by_id = {r["item_id"]: r["content"] for r in response.get("resolved", [])}

        for item in items:
            new_content = resolved_by_id.get(item["item_id"])
            if not new_content or new_content == item["content"]:
                still_unresolved.append(item["item_id"])
                continue
            chunk = chunks_by_id.get(item["chunk_id"])
            target = chunk[item["layer"]][item["index_in_layer"]]
            target["content"] = new_content
            target["needs_lookback"] = False
            resolved_count += 1

    return LookbackSummary(
        initiated=True,
        flagged_count=flagged_count,
        llm_calls=len(groups),
        resolved_count=resolved_count,
        still_unresolved=still_unresolved,
    )

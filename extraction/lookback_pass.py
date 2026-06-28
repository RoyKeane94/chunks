"""
Lookback resolution -- a single, simple LLM call per chunk.

Run after the main per-chunk extraction pass. Chunks must be processed in
ascending chunk_index order: chunk N's lookback cannot run until chunks
N-1 and N-2 are finalised (their own lookback turn complete, whatever the
outcome). Prior context is raw transcript plus all propositions from those
finalised predecessors -- resolved or still flagged, no filtering.

No registry, no alias tracking. If the model rewrites a sentence, we trust
it and substitute. If it returns the sentence unchanged, the item stays flagged.
"""

import json
from dataclasses import dataclass, field

LOOKBACK_SYSTEM_PROMPT = """You will be given the text of the previous chunk(s) of a podcast transcript, plus a list of sentences from the chunk right after it. Each sentence contains an unclear reference (a bare pronoun like "he"/"they," an initial like "S.," a vague label like "the man," or a second-person "you"/"your" with no name attached) that couldn't be resolved without more context.

For third-person references ("he"/"she"/"they"/an initial/a vague label): find the specific moment in the previous chunk(s) that this sentence is continuing -- match the action, event, or detail described (e.g. "drove a coal ship," "went to the Naval Academy," "had a near-miss at Ellis Island") to where that same action or detail is described or set up in the prior text, and identify whose story that is. Do NOT simply pick whichever person's name appears most often or most recently in the prior chunk(s) -- a transcript can mention several different people, and the right answer is whoever the SPECIFIC event in this sentence belongs to, not the most prominent name nearby.

For second-person references ("you"/"your"): resolve to the person being spoken to, not the person speaking. The name after a "**Name:**" tag is who is talking — never use that name as the answer for "you."

If multiple people are mentioned in the prior chunks and you are not sure which one a sentence's events or address belongs to, return the sentence unchanged rather than guessing.

Once you've identified the right person, rewrite the sentence with the unclear reference replaced by their specific name. If you still can't tell even with this extra context, return the sentence unchanged rather than guessing.

Return ONLY valid JSON, no preamble, no markdown fences:
{"resolved": [{"item_id": "...", "content": "..."}]}"""

LOOKBACK_USER_TEMPLATE = """PRIOR CONTEXT (transcript + propositions from earlier chunks):
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
    still_unresolved: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "initiated": self.initiated,
            "flagged_count": self.flagged_count,
            "llm_calls": self.llm_calls,
            "resolved_count": self.resolved_count,
            "still_unresolved_count": len(self.still_unresolved),
            "still_unresolved": self.still_unresolved,
            "calls": self.calls,
        }

    def merge(self, other: "LookbackSummary") -> "LookbackSummary":
        if not other.initiated:
            return self
        if not self.initiated:
            return other
        return LookbackSummary(
            initiated=True,
            flagged_count=self.flagged_count + other.flagged_count,
            llm_calls=self.llm_calls + other.llm_calls,
            resolved_count=self.resolved_count + other.resolved_count,
            still_unresolved=self.still_unresolved + other.still_unresolved,
            calls=self.calls + other.calls,
        )


def find_flagged_items(extracted_chunks):
    """Collect every proposition/claim with needs_lookback=True."""
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


def resolve_chunk_lookback(extracted_chunk, flagged_items, call_llm, prior_text):
    """One lookback LLM call for all flagged items in a single chunk.

    Mutates extracted_chunk in place. Returns a LookbackSummary for this chunk.
    """
    if not flagged_items:
        return LookbackSummary(initiated=False, flagged_count=0)

    items_payload = [{"item_id": it["item_id"], "content": it["content"]} for it in flagged_items]
    user_prompt = LOOKBACK_USER_TEMPLATE.format(
        prior_chunks_text=prior_text,
        items_json=json.dumps(items_payload, ensure_ascii=False, indent=2),
    )
    response = call_llm(LOOKBACK_SYSTEM_PROMPT, user_prompt)
    resolved_by_id = {r["item_id"]: r["content"] for r in response.get("resolved", [])}

    resolved_count = 0
    still_unresolved = []
    for item in flagged_items:
        new_content = resolved_by_id.get(item["item_id"])
        if not new_content or new_content == item["content"]:
            still_unresolved.append(item["item_id"])
            continue
        layer_list = extracted_chunk[item["layer"]]
        layer_list[item["index_in_layer"]]["content"] = new_content
        layer_list[item["index_in_layer"]]["needs_lookback"] = False
        resolved_count += 1

    return LookbackSummary(
        initiated=True,
        flagged_count=len(flagged_items),
        llm_calls=1,
        resolved_count=resolved_count,
        still_unresolved=still_unresolved,
    )


def run_lookback_pass(
    extracted_chunks,
    call_llm,
    get_prior_chunks_text,
    max_lookback_chunks=2,
):
    """Resolve flagged items chunk-by-chunk in ascending chunk_index order.

    get_prior_chunks_text: callable(chunk_index, n) -> str
        Must reflect only finalised predecessor chunks (caller enforces ordering).

    Returns LookbackSummary (initiated=False when nothing was flagged).
    """
    flagged_items = find_flagged_items(extracted_chunks)
    flagged_count = len(flagged_items)
    if not flagged_count:
        return LookbackSummary(initiated=False, flagged_count=0)

    chunks_by_id = {c["chunk_id"]: c for c in extracted_chunks}
    groups = group_by_chunk(flagged_items)
    summary = LookbackSummary(initiated=False, flagged_count=0)

    for chunk_index in sorted(groups.keys()):
        items = groups[chunk_index]
        prior_text = get_prior_chunks_text(chunk_index, max_lookback_chunks)
        chunk_data = chunks_by_id[items[0]["chunk_id"]]
        chunk_summary = resolve_chunk_lookback(chunk_data, items, call_llm, prior_text)
        summary = summary.merge(chunk_summary)

    summary.flagged_count = flagged_count
    return summary
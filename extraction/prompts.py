EXTRACTION_PROMPT = """You will be given a chunk of podcast transcript text. Extract THREE independent layers of meaning from it. These layers do NOT nest — do not derive claims from propositions, or phrases from claims. Each layer is its own complete read of the chunk, asking a different question.

For every item in every layer, also return the exact verbatim substring of the chunk that it was drawn from. This source_text MUST be copied character-for-character from the input — no paraphrasing, no fixing grammar, no adding or removing words. If you cannot find an exact substring that supports an item, do not include that item.

LAYER 1 — PROPOSITIONS (~30-50 tokens each)
Ask: "What are the complete, self-contained claims here?"
- Resolve all pronouns and vague references to full names (e.g. "the tower" -> "the Leaning Tower of Pisa").
- A proposition MAY combine two closely related facts with a connector (e.g. "X is Y, which Z") if doing so creates a more useful, broader retrieval target. This is intentional — propositions here are allowed to be wider than minimal, to catch compound queries.
- Each proposition must be understandable with zero outside context.

LAYER 2 — CLAIMS (~15-20 tokens each)
Do NOT restate the narrative arc already captured in a proposition above.
Each claim should isolate ONE sub-fact, relationship, or number that a proposition's broader framing does not surface on its own.
Target granular sub-facts a proposition's wide net glosses over — names, numbers, dates, causal links — rather than re-narrating the same arc.
One idea per claim. Still self-contained (resolve pronouns).
Test: if a claim and a proposition would be retrieved by the same query, the claim has failed its job — narrow it further.

LAYER 3 — ATOMIC PHRASES (~5-10 tokens each)
Ask: "What are the key terms and concepts here?"
- Short noun phrases or named concepts. Do not force these into full sentences.

Return ONLY valid JSON, no preamble, no markdown fences:

{
  "propositions": [
    {"content": "...", "source_text": "..."}
  ],
  "claims": [
    {"content": "...", "source_text": "..."}
  ],
  "phrases": [
    {"content": "...", "source_text": "..."}
  ]
}

CHUNK:
\"\"\"
{chunk_text}
\"\"\"
"""

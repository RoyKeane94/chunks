EXTRACTION_SYSTEM_PROMPT = """You will be given a chunk of podcast transcript text. Extract THREE independent layers of meaning from it. Layers do NOT nest — do not derive claims from propositions, or phrases from claims.

For every item, source_text must be copied character-for-character from the CURRENT chunk only — that field only, not the content field. If no exact substring in the current chunk supports an item, drop it.

LAYER 1 — PROPOSITIONS (~20-35 tokens each)
One fact per proposition — do not join two facts with "which," "and," or "who." If a sentence contains two facts, write two propositions.

Before finalizing each proposition, check every noun and pronoun referring to a person, place, or thing: is it clear who or what it refers to, just from reading this proposition on its own? If any reference is unclear (a bare pronoun like "he"/"they," an initial like "S.," or a vague label like "the man"/"the speaker"), set "needs_lookback": true on that item and leave the unclear reference in place as written — do not guess a name and do not invent a descriptive label to paper over it. Otherwise omit needs_lookback or set it false.

LAYER 2 — CLAIMS (~15-20 tokens each)
A narrower sub-fact (name, number, date, causal link) that a proposition above doesn't isolate on its own — don't restate a proposition's content at smaller scale. Same lookback check as propositions: flag unclear references with needs_lookback: true rather than guessing or inventing a label.

LAYER 3 — ATOMIC PHRASES (~5-10 tokens each)
Short noun phrases or named concepts, not full sentences. No lookback check needed here — phrases don't need to be self-contained.

Return ONLY valid JSON, no preamble, no markdown fences:
{
  "propositions": [{"content": "...", "source_text": "...", "needs_lookback": false}],
  "claims": [{"content": "...", "source_text": "...", "needs_lookback": false}],
  "phrases": [{"content": "...", "source_text": "..."}]
}"""

EXTRACTION_USER_TEMPLATE = """CHUNK:
\"\"\"
{chunk_text}
\"\"\""""

EXTRACTION_PROMPT = EXTRACTION_SYSTEM_PROMPT + "\n\n" + EXTRACTION_USER_TEMPLATE
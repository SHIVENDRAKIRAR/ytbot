"""groq_script.py — Generate YouTube Short script + image prompts via Groq LLM.

Two modes, one entry point (generate_short_pack):
  • Single-language  → {full_narration, youtube_title, youtube_description, image_prompts}
  • Multi-variant    → {image_prompts, variants: {lang: {title, description, full_narration}}}
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from groq import Groq

from pipeline.channel_presets import ChannelPreset
from pipeline.story_history import history_prompt_block

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS  = 3072
MAX_RETRIES = 3

_LANG_LABELS = {"en": "English", "hi": "Hindi (Devanagari script)"}


@dataclass(frozen=True)
class LangTarget:
    """Word-count bounds and the instruction blurb sent to the model."""
    min_words: int
    max_words: int
    blurb: str

    @property
    def range_str(self) -> str:
        return f"{self.min_words}-{self.max_words}"


LANG_TARGETS: dict[str, LangTarget] = {
    "en": LangTarget(
        120, 155,
        "120-155 English words (~40-50 sec); add transitions, examples, "
        "and a closing takeaway — NOT a bullet list",
    ),
    "hi": LangTarget(
        135, 170,
        "135-170 Devanagari Hindi words (~55-70 sec); "
        "full sentences, not headlines",
    ),
}

# Fallback when a language has no entry in LANG_TARGETS
_DEFAULT_MIN_WORDS = 80
_DEFAULT_LANG_TARGET = LangTarget(_DEFAULT_MIN_WORDS, 200, f"≥{_DEFAULT_MIN_WORDS} words")


# ── Public API ────────────────────────────────────────────────────────────────

def generate_short_pack(
    preset: ChannelPreset,
    *,
    topic_hint: str | None = None,
    channel_id: str | None = None,
) -> dict[str, Any]:
    """Generate a complete Short pack (script + image prompts) for *preset*.

    Returns a dict whose shape depends on whether the preset is single- or
    multi-language (see module docstring).
    """
    topic_hint = (topic_hint or os.environ.get("SHORT_TOPIC", "")).strip()

    base_prompt = _build_base_prompt(preset, topic_hint, channel_id)
    n_segments  = preset["segment_count"]
    variants    = preset.get("variants") or []

    if variants:
        return _generate_multivariant(preset, base_prompt, n_segments, variants)
    return _generate_single(preset, base_prompt, n_segments)


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_base_prompt(
    preset: ChannelPreset,
    topic_hint: str,
    channel_id: str | None,
) -> str:
    lines = [
        f"Channel style: {preset['label']}.",
        "Create ONE YouTube Short.",
    ]
    if topic_hint:
        lines.append(f"Topic idea from creator: {topic_hint}")

    if channel_id:
        anti_repeat = history_prompt_block(channel_id)
        if anti_repeat:
            lines.append(anti_repeat)

    return "\n".join(lines) + "\n"


def _single_language_schema(language: str, n_segments: int) -> str:
    """Return the JSON-schema instruction block for single-language presets."""
    target = LANG_TARGETS.get(language, _DEFAULT_LANG_TARGET)
    lang_label = _lang_labels(language)

    if language == "hi":
        narration_field = (
            f'"full_narration": "COMPLETE narration as ONE continuous paragraph '
            f'in Devanagari Hindi. This is what the voice will read aloud. '
            f'Must be {target.blurb}. '
            f'Natural spoken Hindi — no segment markers, no numbering, no English transliteration."'
        )
        strict_rules = (
            f"- LANGUAGE: full_narration, youtube_title, and youtube_description MUST be in Devanagari Hindi.\n"
            f"- image_prompts MUST be in ENGLISH (image model does not understand Hindi).\n"
            f"- WORD COUNT: full_narration MUST contain {target.range_str} Hindi words.\n"
        )
    else:
        narration_field = (
            f'"full_narration": "COMPLETE story/script as one continuous paragraph. '
            f'This is what the voice will read. '
            f'Must be {target.blurb}. Natural narration — no segment breaks, no numbering."'
        )
        strict_rules = (
            f"- full_narration is ONE continuous paragraph, {target.range_str} {lang_label} words.\n"
        )

    return f"""
Return ONLY valid JSON with this shape:
{{
  "youtube_title": "short catchy title, under 90 chars, no hashtags",
  "youtube_description": "2-3 sentences plus optional #Shorts at end",
  {narration_field},
  "image_prompts": [
    "visual description for image 1: setting, subject, action. No style words. No text in image.",
    "visual description for image 2...",
    "..."
  ]
}}

STRICT RULES:
{strict_rules}- "image_prompts" array MUST have exactly {n_segments} entries.
- Each image_prompt matches a different moment/beat in order.
- Image prompts describe visuals only — no narration text, no style words, no quotes.
- The narration must flow naturally as one spoken piece (no "segment 1", "segment 2" etc).
"""


def _multivariant_schema(variants: list, n_segments: int) -> str:
    """Return the JSON-schema instruction block for multi-variant presets."""
    variant_fields = []
    for v in variants:
        lang   = v["lang"]
        target = LANG_TARGETS.get(lang, _DEFAULT_LANG_TARGET)
        label  = _lang_labels(lang)
        variant_fields.append(
            f'    "{lang}": {{\n'
            f'      "youtube_title": "catchy title in {label} (<90 chars, no hashtags)",\n'
            f'      "youtube_description": "2-3 sentences in {label} + optional #Shorts",\n'
            f'      "full_narration": "ONE continuous paragraph in {label}. '
            f'{target.blurb}. Natural spoken narration, no segment markers."\n'
            f'    }}'
        )

    word_targets = "\n".join(
        f"  - {_lang_labels(v['lang'])}: {LANG_TARGETS.get(v['lang'], _DEFAULT_LANG_TARGET).blurb}"
        for v in variants
    )
    lang_keys = ", ".join(f'"{v["lang"]}"' for v in variants)

    return f"""
Return ONLY valid JSON with this shape:
{{
  "image_prompts": [
    "visual description for image 1 — IN ENGLISH ONLY: setting, subject, action. No style words. No text in image.",
    "..."
  ],
  "variants": {{
{chr(10).join(variant_fields)}
  }}
}}

STRICT RULES:
- "image_prompts" MUST have exactly {n_segments} entries, ALL in English.
- "variants" MUST contain keys: {lang_keys}.
- Each variant tells the SAME facts/story written natively in that language (not a literal translation).
- Word-count targets:
{word_targets}
- Narrations are continuous spoken paragraphs — no segment numbers, no headings.
- Titles/descriptions must each be in their own language.
- Before outputting JSON: mentally count words in each full_narration.
  If any are below the minimum, REWRITE that paragraph (same facts, more sentences) until the count is met.
"""


# ── Generation paths ──────────────────────────────────────────────────────────

def _generate_single(preset: ChannelPreset, base_prompt: str, n: int) -> dict[str, Any]:
    language = (preset.get("language") or "en").lower()
    target   = LANG_TARGETS.get(language, _DEFAULT_LANG_TARGET)
    min_words = preset.get("min_words", target.min_words)

    schema_prompt = _single_language_schema(language, n)
    last_error    = ""

    for attempt in range(MAX_RETRIES):
        prompt = base_prompt + schema_prompt + _retry_suffix(attempt, last_error, target)
        data   = _call_groq(preset, prompt, temperature=_temperature(attempt))

        try:
            _validate_single(data, n, language, min_words)
            return data
        except ValueError as exc:
            last_error = str(exc)
            logger.warning("Attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES, last_error)
            if attempt == MAX_RETRIES - 1:
                raise

    raise RuntimeError("Unreachable")  # satisfy type checkers


def _generate_multivariant(
    preset: ChannelPreset,
    base_prompt: str,
    n: int,
    variants: list,
) -> dict[str, Any]:
    schema_prompt = _multivariant_schema(variants, n)
    last_error    = ""

    for attempt in range(MAX_RETRIES + 1):  # multivariant gets one extra try
        prompt = base_prompt + schema_prompt + _retry_suffix_multi(attempt, last_error)
        data   = _call_groq(preset, prompt, temperature=_temperature(attempt))

        try:
            _validate_multivariant(data, variants, n)
            return data
        except ValueError as exc:
            last_error = str(exc)
            logger.warning("Attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES + 1, last_error)
            if attempt == MAX_RETRIES:
                raise

    raise RuntimeError("Unreachable")


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_single(
    data: dict[str, Any],
    n: int,
    language: str,
    min_words: int,
) -> None:
    narration = (data.get("full_narration") or "").strip()
    if not narration:
        raise ValueError("Missing full_narration")

    _validate_image_prompts(data.get("image_prompts"), n)

    word_count = len(narration.split())
    if word_count < min_words:
        raise ValueError(
            f"Narration too short ({word_count} words, expected ≥{min_words} for {language})"
        )


def _validate_multivariant(data: dict[str, Any], variants: list, n: int) -> None:
    _validate_image_prompts(data.get("image_prompts"), n)

    vmap = data.get("variants")
    if not isinstance(vmap, dict):
        raise ValueError("Response missing 'variants' object")

    for v in variants:
        lang = v["lang"]
        node = vmap.get(lang)
        if not isinstance(node, dict):
            raise ValueError(f"variants['{lang}'] missing")

        narration = (node.get("full_narration") or "").strip()
        if not narration:
            raise ValueError(f"variants['{lang}'].full_narration empty")

        target    = LANG_TARGETS.get(lang, _DEFAULT_LANG_TARGET)
        min_words = v.get("min_words", target.min_words)
        word_count = len(narration.split())
        if word_count < min_words:
            raise ValueError(
                f"variants['{lang}'].full_narration too short "
                f"({word_count} words, need ≥{min_words}; ideal {target.range_str})"
            )

        if not (node.get("youtube_title") or "").strip():
            raise ValueError(f"variants['{lang}'].youtube_title empty")


def _validate_image_prompts(prompts: Any, expected: int) -> None:
    if not isinstance(prompts, list) or len(prompts) != expected:
        raise ValueError(f"Expected {expected} image_prompts, got {len(prompts or [])}")
    for i, p in enumerate(prompts):
        if not isinstance(p, str) or not p.strip():
            raise ValueError(f"image_prompt[{i}] is empty")


# ── Groq client ───────────────────────────────────────────────────────────────

def _call_groq(
    preset: ChannelPreset,
    user_prompt: str,
    *,
    temperature: float = 0.85,
) -> dict[str, Any]:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": preset["groq_system_hint"]},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    if not raw:
        raise RuntimeError("Empty Groq response")
    return json.loads(raw)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lang_labels(lang: str) -> str:
    return _LANG_LABELS.get(lang, lang)


def _temperature(attempt: int) -> float:
    """Higher temp for early attempts (creative); lower for later retries (obedient)."""
    return 0.85 if attempt < 2 else 0.45


def _retry_suffix(attempt: int, last_error: str, target: LangTarget) -> str:
    if attempt == 0 or not last_error:
        return ""
    return (
        f"\n\nCRITICAL — previous attempt failed: {last_error}\n"
        f"Rewrite the narration to be longer and more detailed. "
        f"Aim for {target.range_str} words. "
        f"Add more descriptive sentences to each beat.\n"
    )


def _retry_suffix_multi(attempt: int, last_error: str) -> str:
    if attempt == 0 or not last_error:
        return ""
    return (
        f"\n\n=== REGENERATE — previous JSON failed validation ===\n"
        f"Error: {last_error}\n"
        f"Return a NEW complete JSON object fixing the issue. "
        f"Keep the same facts/story and image_prompts beats; "
        f"expand ONLY the narration(s) that were too short — add 3-5 full sentences each.\n"
    )
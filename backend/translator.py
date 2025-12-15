"""
Translation Service for Torah AI

Uses mistralai/mistral-small-3.2-24b-instruct via OpenRouter
for high-quality, consistent translations of Hebrew Torah texts.
"""

import os
import logging
import httpx
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Translation model configuration
TRANSLATION_MODEL = "mistralai/mistral-small-3.2-24b-instruct"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class TranslationRequest:
    """Request for translating a Torah source."""
    hebrew_text: str
    hebrew_excerpt: Optional[str] = None  # If provided, only translate this portion
    source_ref: str = ""
    book: str = ""
    context_text: Optional[str] = None  # AI-generated context for SA texts

    @property
    def is_shulchan_arukh(self) -> bool:
        return "Shulchan Arukh" in self.source_ref or "Shulchan Arukh" in self.book

    @property
    def is_excerpt(self) -> bool:
        return self.hebrew_excerpt is not None and len(self.hebrew_excerpt.strip()) > 0


@dataclass
class TranslationResult:
    """Result from translation service."""
    translation: str
    success: bool
    error: Optional[str] = None


def build_translation_prompt(request: TranslationRequest) -> tuple[str, str]:
    """
    Build system and user prompts for translation.
    Returns (system_prompt, user_prompt).
    """

    # Base system prompt with core translation rules
    system_prompt = """You are a precise translator of Hebrew Torah texts into English.

CRITICAL OUTPUT RULES:
- Output ONLY the English translation
- NO preamble, introduction, or explanation
- NO phrases like "Here is the translation:" or "The text says:"
- NO markdown formatting, headers, or bullet points
- Just the clean English translation text

TRANSLATION STYLE:
- Clear, readable modern English
- Preserve technical halachic terms in transliteration when standard (e.g., "brachos", "mitzvah")
- Maintain the legal/formal tone appropriate for halachic texts
- Be accurate to the Hebrew meaning"""

    # Add Shulchan Arukh-specific rules
    if request.is_shulchan_arukh:
        system_prompt += """

SHULCHAN ARUKH SPECIFIC RULES:
1. When you see "הגה" (gloss marker), translate it as "Rama:" followed by the Rama's gloss content
2. Source quotes in parentheses: When the text contains parenthetical Hebrew source references (like קידושין ל:), DO NOT translate these - simply omit them or note "[source citation]"
3. Maintain the distinction between the Mechaber (main author) and Rama's additions
4. Use consistent terminology for halachic concepts throughout"""

    # Build user prompt
    if request.is_excerpt:
        user_prompt = f"""Translate ONLY the highlighted portion of this Hebrew text.

FULL SOURCE TEXT (for context):
{request.hebrew_text}

HIGHLIGHTED PORTION TO TRANSLATE:
{request.hebrew_excerpt}

INSTRUCTIONS:
- Translate ONLY the highlighted portion above
- The full text is provided only for context
- Output just the English translation, nothing else"""
    else:
        user_prompt = f"""Translate this Hebrew text:

{request.hebrew_text}

Output just the English translation, nothing else."""

    # Add context snippet if available (for SA texts)
    if request.context_text and request.is_shulchan_arukh:
        user_prompt = f"""Translate this Hebrew text:

{request.hebrew_text}

CONTEXT NOTE (AI-generated summary for reference only - do NOT translate this):
{request.context_text}

Translate the Hebrew text above. Output just the English translation, nothing else."""

    return system_prompt, user_prompt


async def translate_source(
    request: TranslationRequest,
    openrouter_api_key: str,
    timeout: float = 30.0
) -> TranslationResult:
    """
    Translate a Torah source using Mistral Small via OpenRouter.

    This is a NON-STREAMING call that returns the complete translation.
    """
    system_prompt, user_prompt = build_translation_prompt(request)

    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://kesher.ai",
        "X-Title": "Kesher AI Translation"
    }

    payload = {
        "model": TRANSLATION_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,  # Low temp for consistent translations
        "max_tokens": 2000,
        "stream": False,  # Non-streaming for clean output
        "provider": {
            "order": ["mistral"]  # Prioritize Mistral provider
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()

            data = response.json()
            translation = data["choices"][0]["message"]["content"].strip()

            # Clean any remaining preamble patterns
            translation = clean_translation_output(translation)

            return TranslationResult(
                translation=translation,
                success=True
            )

    except httpx.TimeoutException:
        logger.error(f"Translation timeout for {request.source_ref}")
        return TranslationResult(
            translation="",
            success=False,
            error="Translation request timed out"
        )
    except Exception as e:
        logger.error(f"Translation error for {request.source_ref}: {e}")
        return TranslationResult(
            translation="",
            success=False,
            error=str(e)
        )


def clean_translation_output(text: str) -> str:
    """
    Remove any preamble or postamble that the model might add.
    """
    # Common preamble patterns to remove
    preamble_patterns = [
        "Here is the translation:",
        "Here's the translation:",
        "The translation is:",
        "Translation:",
        "English translation:",
    ]

    for pattern in preamble_patterns:
        if text.lower().startswith(pattern.lower()):
            text = text[len(pattern):].strip()

    # Remove markdown code blocks if present
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()

    return text.strip()


async def translate_source_batch(
    requests: list[TranslationRequest],
    openrouter_api_key: str,
    max_concurrent: int = 3
) -> list[TranslationResult]:
    """
    Translate multiple sources concurrently.

    Args:
        requests: List of translation requests
        openrouter_api_key: API key
        max_concurrent: Max concurrent translation calls

    Returns:
        List of translation results in same order as requests
    """
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrent)

    async def translate_with_semaphore(req: TranslationRequest) -> TranslationResult:
        async with semaphore:
            return await translate_source(req, openrouter_api_key)

    tasks = [translate_with_semaphore(req) for req in requests]
    return await asyncio.gather(*tasks)

import logging
import os
import re
from functools import lru_cache
from html import unescape

from dotenv import load_dotenv
from supabase import Client, create_client

from config_commentaries import MAIN_COMMENTATORS

logger = logging.getLogger(__name__)

load_dotenv()

COMMENTARY_REF_NUMBER_RE = re.compile(r"\d+")
HTML_TAG_RE = re.compile(r"<[^>]+>")


@lru_cache(maxsize=1)
def get_supabase_client() -> Client | None:
    """Create a cached Supabase client for commentary hydration."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        logger.error("❌ Missing SUPABASE_URL or SUPABASE_KEY; commentary hydration unavailable")
        return None

    try:
        return create_client(url, key)
    except Exception as e:
        logger.error(f"❌ Failed to initialize Supabase client for commentary hydration: {e}")
        return None


def get_allowed_commentators(book_title: str) -> list[str]:
    """Return configured commentator names for a base book."""
    for key, val in MAIN_COMMENTATORS.items():
        if book_title.startswith(key):
            return val
    return []


def strip_html(text: str) -> str:
    """Remove inline HTML tags stored in cached commentary text."""
    return unescape(HTML_TAG_RE.sub("", text or "")).strip()


def commentary_sort_key(commentary: dict, allowed: list[str]) -> tuple:
    """Sort by configured commentator order, then natural numeric ref order."""
    commentator = commentary.get("commentator", "")
    commentary_ref = commentary.get("ref", "")
    ref_numbers = tuple(int(part) for part in COMMENTARY_REF_NUMBER_RE.findall(commentary_ref))
    commentator_index = allowed.index(commentator) if commentator in allowed else len(allowed)
    return commentator_index, ref_numbers, commentary_ref


def fetch_commentaries_for_ref(sefaria_ref, book_title):
    """
    Fetch cached commentaries from Supabase and filter for configured meforshim.

    Args:
        sefaria_ref: Reference string (e.g., "Shulchan Arukh, Orach Chayim 1:1")
        book_title: Book title from metadata (e.g., "Shulchan Arukh, Orach Chayim")

    Returns:
        List of dicts with 'commentator' and 'text' keys
    """
    logger.debug(f"🔍 Fetching cached commentaries for: {sefaria_ref}")

    allowed = get_allowed_commentators(book_title)
    if not allowed:
        logger.warning(f"⚠️ No configured meforshim found for book: {book_title}")
        return []

    supabase = get_supabase_client()
    if supabase is None:
        return []

    try:
        response = (
            supabase.table("sefaria_commentaries")
            .select("commentator,text_he,commentary_ref")
            .eq("base_ref", sefaria_ref)
            .execute()
        )
        raw_comments = response.data or []
        logger.debug(f"📊 Found {len(raw_comments)} cached commentaries for {sefaria_ref}")
    except Exception as e:
        logger.error(f"❌ Supabase query error for {sefaria_ref}: {e}")
        return []

    filtered = []
    for c in raw_comments:
        title = c.get("commentator", "")
        logger.debug(f"   Checking commentary: {title}")

        if title in allowed:
            he_text = strip_html(c.get("text_he", ""))
            if he_text:
                filtered.append({
                    "commentator": title,
                    "text": he_text,
                    "ref": c.get("commentary_ref", ""),
                })
                logger.debug(
                    f"   ✅ Included cached commentary: {title} ref={c.get('commentary_ref', '')} ({len(he_text)} chars)"
                )
            else:
                logger.debug(f"   ⚠️ Skipped {title} - no Hebrew text")
        else:
            logger.debug(f"   ❌ Not in allowed list: {title}")

    filtered.sort(key=lambda x: commentary_sort_key(x, allowed))

    logger.info(f"✅ Cached hydration complete for {sefaria_ref}: {len(filtered)} commentaries found")
    return filtered
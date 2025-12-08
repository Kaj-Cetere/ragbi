import requests
import logging
from config_commentaries import MAIN_COMMENTATORS

logger = logging.getLogger(__name__)

def fetch_commentaries_for_ref(sefaria_ref, book_title):
    """
    Fetches commentaries from Sefaria API and filters for configured meforshim.
    
    Args:
        sefaria_ref: Reference string (e.g., "Shulchan Arukh, Orach Chayim 1:1")
        book_title: Book title from metadata (e.g., "Shulchan Arukh, Orach Chayim")
    
    Returns:
        List of dicts with 'commentator' and 'text' keys
    """
    url = f"https://www.sefaria.org/api/texts/{sefaria_ref}?commentary=1&context=0"
    
    logger.debug(f"🔍 Fetching commentaries for: {sefaria_ref}")
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            logger.warning(f"⚠️ API returned status {response.status_code} for {sefaria_ref}")
            return []
        data = response.json()
        logger.debug(f"✅ Successfully fetched API response for {sefaria_ref}")
    except requests.exceptions.Timeout:
        logger.error(f"❌ Timeout fetching {sefaria_ref}")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Request error for {sefaria_ref}: {e}")
        return []
    except ValueError as e:
        logger.error(f"❌ JSON decode error for {sefaria_ref}: {e}")
        return []

    raw_comments = data.get("commentary", [])
    logger.debug(f"📊 Found {len(raw_comments)} total commentaries in response")
    
    # Find allowed list for this book
    allowed = []
    for key, val in MAIN_COMMENTATORS.items():
        if book_title.startswith(key):
            allowed = val
            logger.debug(f"📖 Matched book '{book_title}' to config key '{key}'")
            logger.debug(f"   Allowed meforshim: {allowed}")
            break
            
    if not allowed:
        logger.warning(f"⚠️ No configured meforshim found for book: {book_title}")
        return []

    filtered = []
    for c in raw_comments:
        # Handle collectiveTitle which can be a dict or string
        collective_title = c.get("collectiveTitle")
        if isinstance(collective_title, dict):
            title = collective_title.get("en", "")
        else:
            title = collective_title or ""
        
        logger.debug(f"   Checking commentary: {title}")
        
        if title in allowed:
            # Clean HTML from Hebrew text
            he_text = c.get("he", "").replace("<b>","").replace("</b>","")
            if he_text:
                filtered.append({
                    "commentator": title,
                    "text": he_text,
                    "ref": c.get("ref", "")  # Include the actual Sefaria ref
                })
                logger.debug(f"   ✅ Included: {title} ref={c.get('ref', '')} ({len(he_text)} chars)")
            else:
                logger.debug(f"   ⚠️ Skipped {title} - no Hebrew text")
        else:
            logger.debug(f"   ❌ Not in allowed list: {title}")
            
    logger.info(f"✅ Hydration complete for {sefaria_ref}: {len(filtered)} commentaries found")
    
    # Sort by order defined in config
    filtered.sort(key=lambda x: allowed.index(x['commentator']) if x['commentator'] in allowed else 99)
    
    return filtered
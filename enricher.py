import requests
from config_commentaries import MAIN_COMMENTATORS

def fetch_commentaries_for_ref(sefaria_ref, book_title):
    """
    Fetches context from Sefaria API and filters for 'Main Two' commentaries.
    """
    url = f"https://www.sefaria.org/api/texts/{sefaria_ref}?commentary=1&context=0"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200: return []
        data = response.json()
    except:
        return []

    raw_comments = data.get("commentary", [])
    
    # Find allowed list for this book
    allowed = []
    for key, val in MAIN_COMMENTATORS.items():
        if book_title.startswith(key):
            allowed = val
            break
            
    if not allowed: return []

    filtered = []
    for c in raw_comments:
        title = c.get("collectiveTitle")
        if title in allowed:
            # Clean HTML from Hebrew text
            he_text = c.get("he", "").replace("<b>","").replace("</b>","")
            filtered.append({
                "commentator": title,
                "text": he_text
            })
            
    # Sort by order defined in config
    filtered.sort(key=lambda x: allowed.index(x['commentator']) if x['commentator'] in allowed else 99)
    
    return filtered
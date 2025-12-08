# exact "collectiveTitle" keys from Sefaria
MAIN_COMMENTATORS = {
    # --- ORACH CHAYIM ---
    # Mishnah Berurah's collectiveTitle is just "Mishnah Berurah"
    "Shulchan Arukh, Orach Chayim": ["Mishnah Berurah"],
    
    # --- YOREH DEAH ---
    # Ba'er Hetev commentaries across Shulchan Arukh sections share the collectiveTitle "Ba'er Hetev"
    "Shulchan Arukh, Yoreh De'ah": ["Ba'er Hetev"],
    
    # --- CHOSHEN MISHPAT ---
    "Shulchan Arukh, Choshen Mishpat": ["Ba'er Hetev"],
    
    # --- EVEN HAEZER ---
    "Shulchan Arukh, Even HaEzer": ["Ba'er Hetev"],
    
    # --- TALMUD (Generic fallback) ---
    "Talmud": ["Rashi", "Tosafot"] 
}
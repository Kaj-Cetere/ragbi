#!/usr/bin/env python3
"""
Debug Even HaEzer processing
"""

import json
from pathlib import Path

def debug_even_haezer():
    """Debug the Even HaEzer file structure"""
    
    file_path = Path("Shulchan-Arukh-Extracted/Halakhah/Shulchan Arukh/Shulchan Arukh, Even HaEzer/Hebrew/merged.json")
    
    print(f"Reading {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Keys: {list(data.keys())}")
    print(f"Text type: {type(data.get('text'))}")
    
    text = data.get('text')
    
    if isinstance(text, dict):
        print(f"Text is dict with keys: {list(text.keys())}")
        for key, value in text.items():
            print(f"  Key '{key}' type: {type(value)}")
            if isinstance(value, list):
                print(f"    Length: {len(value)}")
                if len(value) > 0:
                    print(f"    First item type: {type(value[0])}")
                    if isinstance(value[0], list):
                        print(f"    First item length: {len(value[0])}")
                        if len(value[0]) > 0:
                            print(f"    First segment type: {type(value[0][0])}")
                            print(f"    First segment preview: {str(value[0][0])[:100]}...")
    elif isinstance(text, list):
        print(f"Text is list with length: {len(text)}")
        if len(text) > 0:
            print(f"First item type: {type(text[0])}")

if __name__ == "__main__":
    debug_even_haezer()

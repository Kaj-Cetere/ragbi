#!/usr/bin/env python3
"""
Clean Shulchan Arukh JSON files for vector database embedding
Processes all four sections: Orach Chayim, Yoreh De'ah, Even HaEzer, Choshen Mishpat
"""

import re
import json
import os
from pathlib import Path
from typing import List, Dict, Any

def clean_sefaria_text(raw_text):
    """Clean HTML and commentary markers from Sefaria text"""
    if not isinstance(raw_text, str):
        return ""
    
    # 1. Remove the Commentary Markers (e.g., <i data-commentator="Shach"></i>)
    # We replace them with an empty string because they contain no text.
    text = re.sub(r'<i data-commentator[^>]+></i>', '', raw_text)
    
    # 2. Replace <br> with a space so words don't stick together
    text = text.replace('<br>', ' ')
    
    # 3. Remove all other HTML tags (<b>, <small>, <span>)
    # This KEEPS the text inside them (like the Rema inside <small>), just removes the styling.
    text = re.sub(r'<[^>]+>', '', text)
    
    # 4. Collapse multiple spaces into one
    text = ' '.join(text.split())
    
    return text

def process_text_segment(text_array) -> List[str]:
    """Process nested text array into clean segments"""
    clean_segments = []
    
    # Handle different structures
    if isinstance(text_array, dict):
        # Even HaEzer structure: {"": [...], "Seder HaGet": [...], "Seder Halitzah": [...]}
        for key, chapters in text_array.items():
            section_name = key if key else "Main"
            print(f"    Processing section '{section_name}' with {len(chapters)} items")
            
            if isinstance(chapters, list):
                for i, chapter in enumerate(chapters):
                    if isinstance(chapter, list):
                        for segment in chapter:
                            if isinstance(segment, str):
                                cleaned = clean_sefaria_text(segment)
                                if cleaned.strip():
                                    clean_segments.append(cleaned)
                    elif isinstance(chapter, str):
                        cleaned = clean_sefaria_text(chapter)
                        if cleaned.strip():
                            clean_segments.append(cleaned)
    elif isinstance(text_array, list):
        # Standard structure: [[...], [...], ...]
        print(f"    Processing standard structure with {len(text_array)} chapters")
        for chapter in text_array:
            if isinstance(chapter, list):
                for segment in chapter:
                    if isinstance(segment, str):
                        cleaned = clean_sefaria_text(segment)
                        if cleaned.strip():
                            clean_segments.append(cleaned)
            elif isinstance(chapter, str):
                cleaned = clean_sefaria_text(chapter)
                if cleaned.strip():
                    clean_segments.append(cleaned)
    
    return clean_segments

def create_clean_document(data: Dict[str, Any], section_name: str) -> Dict[str, Any]:
    """Create a clean document structure for vector database"""
    
    # Process the text content
    clean_segments = process_text_segment(data.get('text', []))
    
    # Create metadata
    metadata = {
        'title': data.get('title', ''),
        'language': data.get('language', ''),
        'section': section_name,
        'version': data.get('versionTitle', 'merged'),
        'source': data.get('versionSource', ''),
        'total_segments': len(clean_segments),
        'original_structure': 'dict' if isinstance(data.get('text'), dict) else ('list' if isinstance(data.get('text'), list) else 'unknown')
    }
    
    # Combine all segments into one document with separators
    full_text = '\n\n---\n\n'.join(clean_segments)
    
    # Create the clean document
    clean_doc = {
        'metadata': metadata,
        'text': full_text,
        'segments': clean_segments
    }
    
    return clean_doc

def process_shulchan_section(section_path: Path, output_path: Path) -> bool:
    """Process a single Shulchan Arukh section"""
    try:
        print(f"Processing {section_path.name}...")
        
        # Read the merged.json file
        merged_file = section_path / 'Hebrew' / 'merged.json'
        if not merged_file.exists():
            print(f"  Warning: {merged_file} not found")
            return False
        
        print(f"  Reading {merged_file}")
        with open(merged_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"  Data keys: {list(data.keys())}")
        
        # Extract section name from path
        section_name = section_path.name.replace('Shulchan Arukh, ', '')
        
        # Create clean document
        clean_doc = create_clean_document(data, section_name)
        
        print(f"  Created document with {clean_doc['metadata']['total_segments']} segments")
        
        # Create output directory
        output_dir = output_path / 'cleaned' / section_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned document
        output_file = output_dir / 'cleaned.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(clean_doc, f, ensure_ascii=False, indent=2)
        
        # Also save individual segments for easier chunking
        segments_dir = output_dir / 'segments'
        segments_dir.mkdir(exist_ok=True)
        
        for i, segment in enumerate(clean_doc['segments']):
            segment_file = segments_dir / f'segment_{i:04d}.txt'
            with open(segment_file, 'w', encoding='utf-8') as f:
                f.write(segment)
        
        # Save metadata separately
        metadata_file = output_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(clean_doc['metadata'], f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ Processed {clean_doc['metadata']['total_segments']} segments")
        print(f"  ✓ Saved to {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {section_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Process all Shulchan Arukh sections"""
    
    # Paths
    base_path = Path("Shulchan-Arukh-Extracted/Halakhah/Shulchan Arukh")
    output_path = Path("Shulchan-Arukh-Cleaned")
    
    # The four main sections
    sections = [
        "Shulchan Arukh, Orach Chayim",
        "Shulchan Arukh, Yoreh De'ah", 
        "Shulchan Arukh, Even HaEzer",
        "Shulchan Arukh, Choshen Mishpat"
    ]
    
    print("🧹 Cleaning Shulchan Arukh for Vector Database")
    print("=" * 50)
    
    processed = 0
    failed = 0
    
    for section in sections:
        section_path = base_path / section
        if section_path.exists():
            if process_shulchan_section(section_path, output_path):
                processed += 1
            else:
                failed += 1
        else:
            print(f"  ✗ Section not found: {section}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"✅ Successfully processed: {processed} sections")
    print(f"❌ Failed: {failed} sections")
    print(f"📁 Output directory: {output_path.absolute()}")
    
    # Create a summary file
    summary = {
        'processed_sections': processed,
        'failed_sections': failed,
        'output_structure': {
            'cleaned': {
                'section_name': {
                    'cleaned.json': 'Full cleaned document with metadata',
                    'metadata.json': 'Just the metadata',
                    'segments': {
                        'segment_XXXX.txt': 'Individual text segments'
                    }
                }
            }
        },
        'usage_notes': {
            'for_vector_db': 'Use the segments/ directory for individual chunks',
            'for_full_text': 'Use cleaned.json for complete text',
            'metadata': 'Each segment inherits the document metadata'
        }
    }
    
    with open(output_path / 'processing_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

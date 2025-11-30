#!/usr/bin/env python3
"""
Clean Shulchan Arukh JSON files for vector database embedding
Preserves siman and seif structure for proper source tracing
"""

import re
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

def clean_sefaria_text(raw_text):
    """Clean HTML and commentary markers from Sefaria text"""
    if not isinstance(raw_text, str):
        return ""
    
    # 1. Remove the Commentary Markers (e.g., <i data-commentator="Shach"></i>)
    text = re.sub(r'<i data-commentator[^>]+></i>', '', raw_text)
    
    # 2. Replace <br> with a space so words don't stick together
    text = text.replace('<br>', ' ')
    
    # 3. Remove all other HTML tags (<b>, <small>, <span>)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 4. Collapse multiple spaces into one
    text = ' '.join(text.split())
    
    return text

def extract_siman_number_from_title(title_text):
    """Extract siman number from chapter title like 'דין השכמת הבוקר. ובו ט סעיפים:'"""
    if not isinstance(title_text, str):
        return None
    
    # Look for patterns like "סימן ק" or just numbers before "סעיפים"
    # The title usually contains the siman information
    match = re.search(r'סימן\s*([א-ת]+)', title_text)
    if match:
        return match.group(1)
    
    # Try to extract from the text structure - this is harder
    return None

def process_text_with_metadata(text_array, section_name: str) -> List[Dict[str, Any]]:
    """Process text while preserving siman/seif structure"""
    segments_with_metadata = []
    
    if isinstance(text_array, dict):
        # Even HaEzer structure: {"": [...], "Seder HaGet": [...], "Seder Halitzah": [...]}
        for key, chapters in text_array.items():
            subsection_name = key if key else "Main"
            print(f"    Processing section '{subsection_name}' with {len(chapters)} items")
            
            if isinstance(chapters, list):
                for siman_idx, chapter in enumerate(chapters):
                    if isinstance(chapter, list):
                        # This is a siman with multiple seifim
                        siman_title = ""
                        
                        for seif_idx, segment in enumerate(chapter):
                            if isinstance(segment, str):
                                cleaned = clean_sefaria_text(segment)
                                if cleaned.strip():
                                    # Extract siman number from first segment (usually contains the title)
                                    if seif_idx == 0 and not siman_title:
                                        siman_title = extract_siman_number_from_title(segment)
                                    
                                    metadata = {
                                        'section': section_name,
                                        'subsection': subsection_name,
                                        'siman': siman_idx + 1,  # 1-based indexing
                                        'siman_title': siman_title,
                                        'seif': seif_idx + 1,  # 1-based indexing
                                        'siman_seif_key': f"{section_name}.{subsection_name}.{siman_idx + 1}.{seif_idx + 1}" if subsection_name != "Main" else f"{section_name}.{siman_idx + 1}.{seif_idx + 1}",
                                        'text_type': 'seif'
                                    }
                                    
                                    segments_with_metadata.append({
                                        'metadata': metadata,
                                        'text': cleaned
                                    })
                    elif isinstance(chapter, str):
                        # Single segment siman
                        cleaned = clean_sefaria_text(chapter)
                        if cleaned.strip():
                            metadata = {
                                'section': section_name,
                                'subsection': subsection_name,
                                'siman': siman_idx + 1,
                                'siman_title': None,
                                'seif': 1,
                                'siman_seif_key': f"{section_name}.{subsection_name}.{siman_idx + 1}.1" if subsection_name != "Main" else f"{section_name}.{siman_idx + 1}.1",
                                'text_type': 'siman'
                            }
                            
                            segments_with_metadata.append({
                                'metadata': metadata,
                                'text': cleaned
                            })
    
    elif isinstance(text_array, list):
        # Standard structure: [[...], [...], ...] where each inner list is a siman
        print(f"    Processing standard structure with {len(text_array)} chapters")
        
        for siman_idx, chapter in enumerate(text_array):
            if isinstance(chapter, list):
                # This is a siman with multiple seifim
                siman_title = ""
                
                for seif_idx, segment in enumerate(chapter):
                    if isinstance(segment, str):
                        cleaned = clean_sefaria_text(segment)
                        if cleaned.strip():
                            # Extract siman number from first segment
                            if seif_idx == 0 and not siman_title:
                                siman_title = extract_siman_number_from_title(segment)
                            
                            metadata = {
                                'section': section_name,
                                'subsection': 'Main',
                                'siman': siman_idx + 1,
                                'siman_title': siman_title,
                                'seif': seif_idx + 1,
                                'siman_seif_key': f"{section_name}.{siman_idx + 1}.{seif_idx + 1}",
                                'text_type': 'seif'
                            }
                            
                            segments_with_metadata.append({
                                'metadata': metadata,
                                'text': cleaned
                            })
            elif isinstance(chapter, str):
                # Single segment siman
                cleaned = clean_sefaria_text(chapter)
                if cleaned.strip():
                    metadata = {
                        'section': section_name,
                        'subsection': 'Main',
                        'siman': siman_idx + 1,
                        'siman_title': None,
                        'seif': 1,
                        'siman_seif_key': f"{section_name}.{siman_idx + 1}.1",
                        'text_type': 'siman'
                    }
                    
                    segments_with_metadata.append({
                        'metadata': metadata,
                        'text': cleaned
                    })
    
    return segments_with_metadata

def create_clean_document_with_metadata(data: Dict[str, Any], section_name: str) -> Dict[str, Any]:
    """Create a clean document structure with preserved metadata"""
    
    # Process the text content with metadata
    segments_with_metadata = process_text_with_metadata(data.get('text', []), section_name)
    
    # Create metadata
    document_metadata = {
        'title': data.get('title', ''),
        'language': data.get('language', ''),
        'section': section_name,
        'version': data.get('versionTitle', 'merged'),
        'source': data.get('versionSource', ''),
        'total_segments': len(segments_with_metadata),
        'original_structure': 'dict' if isinstance(data.get('text'), dict) else ('list' if isinstance(data.get('text'), list) else 'unknown'),
        'description': f'Shulchan Arukh {section_name} with preserved siman/seif structure'
    }
    
    # Combine all segments into one document with separators
    full_text = '\n\n---\n\n'.join([seg['text'] for seg in segments_with_metadata])
    
    # Create the clean document
    clean_doc = {
        'document_metadata': document_metadata,
        'text': full_text,
        'segments': segments_with_metadata
    }
    
    return clean_doc

def process_shulchan_section_with_metadata(section_path: Path, output_path: Path) -> bool:
    """Process a single Shulchan Arukh section with metadata preservation"""
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
        
        # Extract section name from path
        section_name = section_path.name.replace('Shulchan Arukh, ', '')
        
        # Create clean document with metadata
        clean_doc = create_clean_document_with_metadata(data, section_name)
        
        print(f"  Created document with {clean_doc['document_metadata']['total_segments']} segments")
        
        # Create output directory
        output_dir = output_path / 'cleaned_with_metadata' / section_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned document
        output_file = output_dir / 'cleaned_with_metadata.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(clean_doc, f, ensure_ascii=False, indent=2)
        
        # Save individual segments with metadata for easier chunking
        segments_dir = output_dir / 'segments'
        segments_dir.mkdir(exist_ok=True)
        
        for i, segment in enumerate(clean_doc['segments']):
            segment_file = segments_dir / f'siman_{segment["metadata"]["siman"]}_seif_{segment["metadata"]["seif"]}.json'
            with open(segment_file, 'w', encoding='utf-8') as f:
                json.dump(segment, f, ensure_ascii=False, indent=2)
        
        # Also save plain text files for compatibility
        text_segments_dir = output_dir / 'text_segments'
        text_segments_dir.mkdir(exist_ok=True)
        
        for i, segment in enumerate(clean_doc['segments']):
            segment_file = text_segments_dir / f'siman_{segment["metadata"]["siman"]}_seif_{segment["metadata"]["seif"]}.txt'
            with open(segment_file, 'w', encoding='utf-8') as f:
                # Add metadata header to text file
                f.write(f"=== {section_name} ===\n")
                f.write(f"Siman: {segment['metadata']['siman']}\n")
                f.write(f"Seif: {segment['metadata']['seif']}\n")
                f.write(f"Key: {segment['metadata']['siman_seif_key']}\n")
                f.write(f"Source: {data.get('versionSource', '')}\n")
                f.write("\n" + "="*50 + "\n\n")
                f.write(segment['text'])
        
        # Save metadata separately
        metadata_file = output_dir / 'document_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(clean_doc['document_metadata'], f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ Processed {clean_doc['document_metadata']['total_segments']} segments")
        print(f"  ✓ Saved to {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {section_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Process all Shulchan Arukh sections with metadata preservation"""
    
    # Paths
    base_path = Path("Shulchan-Arukh-Extracted/Halakhah/Shulchan Arukh")
    output_path = Path("Shulchan-Arukh-Cleaned-With-Metadata")
    
    # The four main sections
    sections = [
        "Shulchan Arukh, Orach Chayim",
        "Shulchan Arukh, Yoreh De'ah", 
        "Shulchan Arukh, Even HaEzer",
        "Shulchan Arukh, Choshen Mishpat"
    ]
    
    print("🧹 Cleaning Shulchan Arukh with Metadata Preservation")
    print("=" * 60)
    
    processed = 0
    failed = 0
    
    for section in sections:
        section_path = base_path / section
        if section_path.exists():
            if process_shulchan_section_with_metadata(section_path, output_path):
                processed += 1
            else:
                failed += 1
        else:
            print(f"  ✗ Section not found: {section}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Successfully processed: {processed} sections")
    print(f"❌ Failed: {failed} sections")
    print(f"📁 Output directory: {output_path.absolute()}")
    
    # Create a summary file
    summary = {
        'processed_sections': processed,
        'failed_sections': failed,
        'output_structure': {
            'cleaned_with_metadata': {
                'section_name': {
                    'cleaned_with_metadata.json': 'Full cleaned document with metadata',
                    'document_metadata.json': 'Document metadata',
                    'segments': {
                        'siman_X_seif_Y.json': 'Individual segments with full metadata'
                    },
                    'text_segments': {
                        'siman_X_seif_Y.txt': 'Plain text with metadata header'
                    }
                }
            }
        },
        'metadata_fields': {
            'section': 'Shulchan Arukh section (Orach Chayim, Yoreh Deah, etc.)',
            'subsection': 'For Even HaEzer: Main, Seder HaGet, Seder Halitzah',
            'siman': 'Chapter number (1-based)',
            'seif': 'Section number within siman (1-based)',
            'siman_seif_key': 'Unique identifier for source tracing',
            'text_type': 'Type: "siman" or "seif"'
        },
        'usage_notes': {
            'for_vector_db': 'Use segments/ JSON files for full metadata',
            'for_simple_text': 'Use text_segments/ TXT files',
            'source_tracing': 'Use siman_seif_key field for traceability',
            'sefaria_urls': f'Construct as: https://www.sefaria.org/Shulchan_Arukh,_{{section}}:{{siman}}:{{seif}}'
        }
    }
    
    with open(output_path / 'processing_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

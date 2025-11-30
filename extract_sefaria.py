#!/usr/bin/env python3
"""
Extract Sefaria files from Git repository without checking out.
This handles Windows filename limitations by sanitizing paths.
"""

import os
import subprocess
import re
from pathlib import Path

def sanitize_path(path):
    """Sanitize a path to be Windows-compatible."""
    # Replace invalid characters
    replacements = {
        '"': '',
        '?': '_q_',
        '*': '_star_',
        '<': '_lt_',
        '>': '_gt_',
        '|': '_pipe_',
        ':': '_colon_',
    }
    
    for old, new in replacements.items():
        path = path.replace(old, new)
    
    # Limit path length (Windows limit is 260, but we'll be conservative)
    if len(path) > 200:
        # Try to preserve the filename by shortening directory names
        parts = path.split('/')
        filename = parts[-1]
        dirs = parts[:-1]
        
        # Shorten each directory if needed
        new_dirs = []
        for d in dirs:
            if len(d) > 20:
                new_dirs.append(d[:17] + '...')
            else:
                new_dirs.append(d)
        
        path = '/'.join(new_dirs + [filename])
    
    return path

def extract_files():
    """Extract all files from Git to a safe location."""
    repo_path = Path("Sefaria-Export")
    output_path = Path("Sefaria-Export-Extracted")
    
    # Get list of all files in the repo
    result = subprocess.run(
        ["git", "ls-tree", "-r", "HEAD", "--name-only"],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error getting file list: {result.stderr}")
        return
    
    files = result.stdout.strip().split('\n')
    print(f"Found {len(files)} files to extract")
    
    extracted = 0
    failed = 0
    
    for file_path in files:
        if not file_path:
            continue
            
        # Skip directories (they end with / in the output)
        if file_path.endswith('/'):
            continue
        
        # Sanitize the path for Windows
        safe_path = sanitize_path(file_path)
        output_file = output_path / safe_path
        
        try:
            # Create directory if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Extract file content (handle both text and binary)
            result = subprocess.run(
                ["git", "show", f"HEAD:{file_path}"],
                cwd=repo_path,
                capture_output=True
            )
            
            if result.returncode == 0:
                # Try to detect if it's binary or text
                content = result.stdout
                try:
                    # Try to decode as UTF-8 text
                    text_content = content.decode('utf-8')
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                except UnicodeDecodeError:
                    # Treat as binary
                    with open(output_file, 'wb') as f:
                        f.write(content)
                extracted += 1
                if extracted % 100 == 0:
                    print(f"Extracted {extracted} files...")
            else:
                print(f"Failed to extract {file_path}: {result.stderr.decode('utf-8', errors='ignore')}")
                failed += 1
                
        except Exception as e:
            print(f"Error extracting {file_path}: {e}")
            failed += 1
    
    print(f"\nExtraction complete!")
    print(f"Successfully extracted: {extracted} files")
    print(f"Failed: {failed} files")
    print(f"Files saved to: {output_path.absolute()}")

if __name__ == "__main__":
    extract_files()

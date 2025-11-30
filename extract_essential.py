#!/usr/bin/env python3
"""
Extract essential Sefaria files with safe filenames
"""

import os
import subprocess
import re
from pathlib import Path

def safe_filename(filename):
    """Convert filename to Windows-safe version"""
    # Replace problematic characters
    filename = re.sub(r'[^\w\s\-_.]', '_', filename)
    # Replace multiple underscores with single
    filename = re.sub(r'_+', '_', filename)
    # Remove trailing/leading underscores
    filename = filename.strip('_')
    # Limit length
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:95] + ext
    return filename

def extract_merged_files():
    """Extract only merged files (the most complete texts)"""
    repo_path = Path("Sefaria-Export")
    output_path = Path("Sefaria-Extracted-Merged")
    
    print("Extracting merged files (most complete texts)...")
    
    # Get all merged files
    result = subprocess.run(
        ["git", "ls-tree", "-r", "HEAD", "--name-only"],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return
    
    files = [f for f in result.stdout.strip().split('\n') if f and 'merged' in f and not f.endswith('/')]
    print(f"Found {len(files)} merged files")
    
    extracted = 0
    failed = 0
    
    for file_path in files:
        try:
            # Create a safe directory structure
            parts = file_path.split('/')
            safe_parts = [safe_filename(p) for p in parts[:-1]]  # Don't sanitize the filename itself yet
            filename = parts[-1]
            
            # Create output path
            output_file = output_path
            for part in safe_parts:
                output_file = output_file / part
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Extract file content
            result = subprocess.run(
                ["git", "show", f"HEAD:{file_path}"],
                cwd=repo_path,
                capture_output=True
            )
            
            if result.returncode == 0:
                content = result.stdout
                try:
                    text_content = content.decode('utf-8')
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                except UnicodeDecodeError:
                    with open(output_file, 'wb') as f:
                        f.write(content)
                extracted += 1
                if extracted % 50 == 0:
                    print(f"  Extracted {extracted} merged files...")
            else:
                failed += 1
                
        except Exception as e:
            print(f"Error with {file_path}: {e}")
            failed += 1
    
    print(f"\nMerged files extraction complete!")
    print(f"Successfully extracted: {extracted} files")
    print(f"Failed: {failed} files")
    print(f"Files saved to: {output_path.absolute()}")

def extract_tanakh_and_talmud():
    """Extract Tanakh and Talmud texts (most commonly used)"""
    repo_path = Path("Sefaria-Export")
    output_path = Path("Sefaria-Extracted-Core")
    
    print("\nExtracting core texts (Tanakh and Talmud)...")
    
    # Focus on main categories
    patterns = [
        "txt/Tanakh/",
        "txt/Talmud/",
        "txt/Mishnah/",
        "json/Tanakh/",
        "json/Talmud/",
        "json/Mishnah/"
    ]
    
    total_extracted = 0
    total_failed = 0
    
    for pattern in patterns:
        print(f"\nExtracting {pattern}...")
        
        # Get files in this pattern
        result = subprocess.run(
            ["git", "ls-tree", "-r", "HEAD", "--name-only", pattern],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error getting {pattern}: {result.stderr}")
            continue
        
        files = [f for f in result.stdout.strip().split('\n') if f and not f.endswith('/')]
        print(f"  Found {len(files)} files")
        
        pattern_extracted = 0
        pattern_failed = 0
        
        for file_path in files[:100]:  # Limit to first 100 files per pattern
            try:
                # Create relative path
                rel_path = file_path.split('/', 1)[-1]  # Remove txt/ or json/ prefix
                output_file = output_path / rel_path
                
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Extract file content
                result = subprocess.run(
                    ["git", "show", f"HEAD:{file_path}"],
                    cwd=repo_path,
                    capture_output=True
                )
                
                if result.returncode == 0:
                    content = result.stdout
                    try:
                        text_content = content.decode('utf-8')
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(text_content)
                    except UnicodeDecodeError:
                        with open(output_file, 'wb') as f:
                            f.write(content)
                    pattern_extracted += 1
                else:
                    pattern_failed += 1
                    
            except Exception as e:
                print(f"    Error with {file_path}: {e}")
                pattern_failed += 1
        
        print(f"  Extracted: {pattern_extracted}, Failed: {pattern_failed}")
        total_extracted += pattern_extracted
        total_failed += pattern_failed
    
    print(f"\nCore texts extraction complete!")
    print(f"Successfully extracted: {total_extracted} files")
    print(f"Failed: {total_failed} files")
    print(f"Files saved to: {output_path.absolute()}")

if __name__ == "__main__":
    extract_merged_files()
    extract_tanakh_and_talmud()

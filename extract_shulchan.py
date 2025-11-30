#!/usr/bin/env python3
"""
Extract Shulchan Arukh from Sefaria repository
"""

import os
import subprocess
from pathlib import Path

def extract_shulchan_arukh():
    """Extract all Shulchan Arukh files"""
    repo_path = Path("Sefaria-Export")
    output_path = Path("Shulchan-Arukh-Extracted")
    
    print("Extracting Shulchan Arukh...")
    
    # Get all Shulchan Arukh files
    result = subprocess.run(
        ["git", "ls-tree", "-r", "HEAD", "--name-only"],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return
    
    # Filter for Shulchan Arukh files
    all_files = result.stdout.strip().split('\n')
    shulchan_files = [f for f in all_files if f and "Shulchan Arukh" in f and not f.endswith('/')]
    
    print(f"Found {len(shulchan_files)} Shulchan Arukh files")
    
    # Also get JSON versions
    json_files = [f for f in all_files if f and "Shulchan_Arukh" in f and not f.endswith('/')]
    print(f"Found {len(json_files)} JSON Shulchan Arukh files")
    
    all_target_files = shulchan_files + json_files
    
    extracted = 0
    failed = 0
    
    for file_path in all_target_files:
        try:
            # Create relative path (remove txt/ or json/ prefix)
            if file_path.startswith('txt/'):
                rel_path = file_path[4:]  # Remove 'txt/'
            elif file_path.startswith('json/'):
                rel_path = file_path[5:]  # Remove 'json/'
            else:
                rel_path = file_path
            
            output_file = output_path / rel_path
            
            # Create directory if needed
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
                    # Try as text first
                    text_content = content.decode('utf-8')
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                except UnicodeDecodeError:
                    # Fallback to binary
                    with open(output_file, 'wb') as f:
                        f.write(content)
                extracted += 1
                if extracted % 20 == 0:
                    print(f"  Extracted {extracted} files...")
            else:
                print(f"Failed to extract: {file_path}")
                failed += 1
                
        except Exception as e:
            print(f"Error with {file_path}: {e}")
            failed += 1
    
    print(f"\nExtraction complete!")
    print(f"Successfully extracted: {extracted} files")
    print(f"Failed: {failed} files")
    print(f"Files saved to: {output_path.absolute()}")
    
    # Show directory structure
    print(f"\nDirectory structure:")
    try:
        for root, dirs, files in os.walk(output_path):
            level = root.replace(str(output_path), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files per directory
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
    except Exception as e:
        print(f"Could not list directory structure: {e}")

if __name__ == "__main__":
    extract_shulchan_arukh()

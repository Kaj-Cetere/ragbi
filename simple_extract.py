#!/usr/bin/env python3
"""
Simple extraction of main Sefaria directories
"""

import os
import subprocess
from pathlib import Path

def extract_directory(git_path, output_dir):
    """Extract all files from a specific directory"""
    repo_path = Path("Sefaria-Export")
    output_path = Path("Sefaria-Export-Extracted") / output_dir
    
    print(f"Extracting {git_path} -> {output_path}")
    
    # Get list of files in this directory
    result = subprocess.run(
        ["git", "ls-tree", "-r", "HEAD", "--name-only", git_path],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error getting file list for {git_path}: {result.stderr}")
        return 0, 0
    
    files = [f for f in result.stdout.strip().split('\n') if f and not f.endswith('/')]
    print(f"Found {len(files)} files in {git_path}")
    
    extracted = 0
    failed = 0
    
    for file_path in files:
        try:
            # Create relative path without the prefix
            rel_path = file_path[len(git_path):].lstrip('/')
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
                # Try to detect if it's binary or text
                try:
                    text_content = content.decode('utf-8')
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                except UnicodeDecodeError:
                    with open(output_file, 'wb') as f:
                        f.write(content)
                extracted += 1
            else:
                print(f"Failed to extract {file_path}")
                failed += 1
                
        except Exception as e:
            print(f"Error extracting {file_path}: {e}")
            failed += 1
    
    print(f"  Extracted: {extracted}, Failed: {failed}")
    return extracted, failed

def main():
    """Extract main content directories"""
    directories = [
        ("json/", "json"),
        ("txt/", "txt"),
        ("xml/", "xml"),
        ("links/", "links"),
        ("schemas/", "schemas"),
    ]
    
    total_extracted = 0
    total_failed = 0
    
    for git_path, output_dir in directories:
        extracted, failed = extract_directory(git_path, output_dir)
        total_extracted += extracted
        total_failed += failed
    
    print(f"\nTotal extraction complete!")
    print(f"Successfully extracted: {total_extracted} files")
    print(f"Failed: {total_failed} files")
    print(f"Files saved to: Sefaria-Export-Extracted/")

if __name__ == "__main__":
    main()

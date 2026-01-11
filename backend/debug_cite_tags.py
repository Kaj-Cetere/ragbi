#!/usr/bin/env python3
"""
Debug script for analyzing LLM cite tag output.

This script:
1. Shows raw LLM output without processing
2. Detects and flags malformed cite tags
3. Highlights what the LLM is doing wrong

Usage:
    # As a module - add to citation_agent.py:
    from debug_cite_tags import analyze_and_log_response

    # Or run standalone with a test response:
    python debug_cite_tags.py
"""

import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TagType(Enum):
    VALID_SELF_CLOSING = "valid_self_closing"
    VALID_SELF_CLOSING_NO_HIGHLIGHT = "valid_self_closing_no_highlight"
    PAIRED_TAG = "paired_tag"
    PAIRED_TAG_WITH_HIGHLIGHT = "paired_tag_with_highlight"
    ORPHANED_CLOSING = "orphaned_closing"
    ORPHANED_OPENING = "orphaned_opening"
    MALFORMED = "malformed"


@dataclass
class DetectedTag:
    tag_type: TagType
    full_match: str
    ref: Optional[str] = None
    highlight: Optional[str] = None
    content: Optional[str] = None
    position: int = 0
    issue: Optional[str] = None


# Color codes for terminal output
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def detect_cite_tags(text: str) -> list[DetectedTag]:
    """
    Detect all cite tag patterns in text, including malformed ones.
    """
    detected = []

    # Pattern 1: Valid self-closing with highlight (CORRECT FORMAT)
    valid_pattern = re.compile(
        r'''<cite\s+ref="([^"]+)"(?:\s+highlight=(?:'([^']*)'|"([^"]*)"))?\s*/>''',
        re.DOTALL
    )

    # Pattern 2: Paired tags <cite ref="...">content</cite>
    paired_pattern = re.compile(
        r'''<cite\s+ref="([^"]+)"(?:\s+highlight=(?:'([^']*)'|"([^"]*)"))?>(.*?)</cite>''',
        re.DOTALL
    )

    # Pattern 3: Orphaned closing tags </cite>
    orphaned_closing = re.compile(r'</cite>')

    # Pattern 4: Opening tags that don't self-close (potential orphans)
    opening_pattern = re.compile(
        r'''<cite\s+ref="([^"]+)"(?:\s+highlight=(?:'([^']*)'|"([^"]*)"))?(?:\s*)>(?!</cite>)''',
        re.DOTALL
    )

    # Track positions of valid matches to avoid double-counting
    matched_ranges = []

    # Find valid self-closing tags
    for match in valid_pattern.finditer(text):
        ref = match.group(1)
        highlight = match.group(2) or match.group(3)
        tag_type = TagType.VALID_SELF_CLOSING if highlight else TagType.VALID_SELF_CLOSING_NO_HIGHLIGHT

        detected.append(DetectedTag(
            tag_type=tag_type,
            full_match=match.group(0),
            ref=ref,
            highlight=highlight,
            position=match.start()
        ))
        matched_ranges.append((match.start(), match.end()))

    # Find paired tags
    for match in paired_pattern.finditer(text):
        # Skip if this overlaps with a valid self-closing tag
        if any(start <= match.start() < end or start < match.end() <= end
               for start, end in matched_ranges):
            continue

        ref = match.group(1)
        highlight = match.group(2) or match.group(3)
        content = match.group(4)

        tag_type = TagType.PAIRED_TAG_WITH_HIGHLIGHT if highlight else TagType.PAIRED_TAG
        issue = "LLM used paired <cite>content</cite> instead of self-closing <cite .../>"
        if highlight and content:
            issue += f" - content '{content[:30]}...' duplicates highlight"

        detected.append(DetectedTag(
            tag_type=tag_type,
            full_match=match.group(0),
            ref=ref,
            highlight=highlight,
            content=content,
            position=match.start(),
            issue=issue
        ))
        matched_ranges.append((match.start(), match.end()))

    # Find orphaned closing tags (not part of a paired tag)
    for match in orphaned_closing.finditer(text):
        # Skip if this is part of a matched pair
        if any(start <= match.start() < end for start, end in matched_ranges):
            continue

        detected.append(DetectedTag(
            tag_type=TagType.ORPHANED_CLOSING,
            full_match=match.group(0),
            position=match.start(),
            issue="Orphaned </cite> tag - will appear as visible text!"
        ))

    # Sort by position
    detected.sort(key=lambda x: x.position)

    return detected


def analyze_response(text: str) -> dict:
    """
    Analyze LLM response and return structured analysis.
    """
    tags = detect_cite_tags(text)

    analysis = {
        "raw_text": text,
        "total_tags": len(tags),
        "valid_tags": sum(1 for t in tags if t.tag_type in [
            TagType.VALID_SELF_CLOSING,
            TagType.VALID_SELF_CLOSING_NO_HIGHLIGHT
        ]),
        "problematic_tags": sum(1 for t in tags if t.tag_type not in [
            TagType.VALID_SELF_CLOSING,
            TagType.VALID_SELF_CLOSING_NO_HIGHLIGHT
        ]),
        "tags": tags,
        "issues": [t.issue for t in tags if t.issue]
    }

    return analysis


def format_analysis_report(analysis: dict, show_raw: bool = True) -> str:
    """
    Format analysis as a readable report.
    """
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"{Colors.BOLD}CITE TAG ANALYSIS REPORT{Colors.RESET}")
    lines.append(f"{'='*80}\n")

    # Summary
    valid = analysis["valid_tags"]
    problematic = analysis["problematic_tags"]
    total = analysis["total_tags"]

    if problematic > 0:
        lines.append(f"{Colors.RED}{Colors.BOLD}[!] ISSUES DETECTED{Colors.RESET}")
    else:
        lines.append(f"{Colors.GREEN}[OK] All tags valid{Colors.RESET}")

    lines.append(f"\nTotal tags found: {total}")
    lines.append(f"  {Colors.GREEN}Valid: {valid}{Colors.RESET}")
    lines.append(f"  {Colors.RED}Problematic: {problematic}{Colors.RESET}")

    # Show raw text with highlighting
    if show_raw:
        lines.append(f"\n{'-'*40}")
        lines.append(f"{Colors.BOLD}RAW LLM OUTPUT:{Colors.RESET}")
        lines.append(f"{'-'*40}")

        # Highlight tags in the raw text
        highlighted_text = analysis["raw_text"]

        # Color code different tag types
        for tag in reversed(analysis["tags"]):  # Reverse to preserve positions
            if tag.tag_type in [TagType.VALID_SELF_CLOSING, TagType.VALID_SELF_CLOSING_NO_HIGHLIGHT]:
                color = Colors.GREEN
            elif tag.tag_type == TagType.ORPHANED_CLOSING:
                color = Colors.RED + Colors.BOLD
            else:
                color = Colors.YELLOW

            start = tag.position
            end = start + len(tag.full_match)
            highlighted_text = (
                highlighted_text[:start] +
                color + tag.full_match + Colors.RESET +
                highlighted_text[end:]
            )

        lines.append(highlighted_text)

    # Detailed tag breakdown
    if analysis["tags"]:
        lines.append(f"\n{'-'*40}")
        lines.append(f"{Colors.BOLD}TAG DETAILS:{Colors.RESET}")
        lines.append(f"{'-'*40}")

        for i, tag in enumerate(analysis["tags"], 1):
            # Icon based on type
            if tag.tag_type in [TagType.VALID_SELF_CLOSING, TagType.VALID_SELF_CLOSING_NO_HIGHLIGHT]:
                icon = f"{Colors.GREEN}[OK]{Colors.RESET}"
            elif tag.tag_type == TagType.ORPHANED_CLOSING:
                icon = f"{Colors.RED}[!!]{Colors.RESET}"
            else:
                icon = f"{Colors.YELLOW}[!!]{Colors.RESET}"

            lines.append(f"\n{icon} Tag #{i}: {tag.tag_type.value}")
            lines.append(f"    Position: {tag.position}")
            lines.append(f"    Match: {Colors.CYAN}{tag.full_match[:100]}{'...' if len(tag.full_match) > 100 else ''}{Colors.RESET}")

            if tag.ref:
                lines.append(f"    Ref: {tag.ref}")
            if tag.highlight:
                lines.append(f"    Highlight: {tag.highlight[:50]}{'...' if len(tag.highlight) > 50 else ''}")
            if tag.content:
                lines.append(f"    Content: {tag.content[:50]}{'...' if len(tag.content) > 50 else ''}")
            if tag.issue:
                lines.append(f"    {Colors.RED}Issue: {tag.issue}{Colors.RESET}")

    # Issues summary
    if analysis["issues"]:
        lines.append(f"\n{'-'*40}")
        lines.append(f"{Colors.RED}{Colors.BOLD}ISSUES FOUND:{Colors.RESET}")
        lines.append(f"{'-'*40}")
        for issue in analysis["issues"]:
            lines.append(f"  {Colors.RED}* {issue}{Colors.RESET}")

    lines.append(f"\n{'='*80}\n")

    return "\n".join(lines)


def analyze_and_log_response(text: str, logger=None):
    """
    Convenience function to analyze and log LLM response.
    Can be called from citation_agent.py to debug.
    """
    analysis = analyze_response(text)
    report = format_analysis_report(analysis)

    if logger:
        logger.info(report)
    else:
        print(report)

    return analysis


# Test examples
TEST_EXAMPLES = [
    # Valid self-closing with highlight
    (
        "The Shulchan Arukh rules <cite ref=\"Shulchan Arukh 208:7\" highlight='בורא פרי האדמה'/> beforehand.",
        "Valid self-closing with highlight"
    ),
    # Valid self-closing without highlight
    (
        "See <cite ref=\"Mishnah Berurah 208:24\"/> for details.",
        "Valid self-closing without highlight"
    ),
    # WRONG: Paired tag (the issue we're debugging)
    (
        "The blessing is <cite ref=\"Shulchan Arukh 208:7\">בורא פרי האדמה</cite> beforehand.",
        "WRONG: Paired tag instead of self-closing"
    ),
    # WRONG: Paired tag with highlight AND content
    (
        "The blessing is <cite ref=\"Shulchan Arukh 208:7\" highlight='בורא פרי האדמה'>בורא פרי האדמה</cite> beforehand.",
        "WRONG: Paired tag with duplicate highlight and content"
    ),
    # WRONG: Orphaned closing tag
    (
        "בורא פרי האדמה</cite> beforehand and",
        "WRONG: Orphaned closing tag (what user is seeing)"
    ),
    # Mixed valid and invalid
    (
        "First <cite ref=\"SA 1:1\" highlight='text'/> then <cite ref=\"SA 1:2\">more text</cite> and orphan</cite>",
        "Mixed: valid + paired + orphan"
    ),
]


def run_tests():
    """Run analysis on test examples."""
    print(f"\n{Colors.BOLD}{'#'*80}")
    print("# CITE TAG DEBUG TESTS")
    print(f"{'#'*80}{Colors.RESET}\n")

    for text, description in TEST_EXAMPLES:
        print(f"\n{Colors.MAGENTA}TEST: {description}{Colors.RESET}")
        print(f"Input: {text[:80]}{'...' if len(text) > 80 else ''}")

        analysis = analyze_response(text)
        print(format_analysis_report(analysis, show_raw=True))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests()
    elif len(sys.argv) > 1:
        # Analyze text from command line
        text = " ".join(sys.argv[1:])
        analysis = analyze_response(text)
        print(format_analysis_report(analysis))
    else:
        # Interactive mode or run tests
        print("Usage:")
        print("  python debug_cite_tags.py --test           # Run test examples")
        print("  python debug_cite_tags.py 'text to analyze'  # Analyze specific text")
        print("\nRunning tests by default...\n")
        run_tests()

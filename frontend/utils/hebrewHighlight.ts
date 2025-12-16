/**
 * Hebrew Text Highlighting Utilities
 *
 * Provides functions for finding and highlighting Hebrew highlights
 * within larger Hebrew text blocks, with smart truncation that
 * ensures the highlighted portion is always visible.
 */

export interface HighlightResult {
  /** Segmented text with highlight markers */
  segments: TextSegment[];
  /** Whether a match was found */
  found: boolean;
  /** Start index of the match in normalized text */
  matchStart?: number;
  /** End index of the match in normalized text */
  matchEnd?: number;
}

export interface TextSegment {
  text: string;
  highlighted: boolean;
}

export interface RenderableSegment {
  key: string;
  text: string;
  highlighted: boolean;
}

/**
 * Normalize Hebrew text for matching
 * Removes HTML tags, extra whitespace, and normalizes Unicode
 */
export function normalizeHebrew(text: string): string {
  if (!text) return "";

  return (
    text
      // Remove HTML tags
      .replace(/<[^>]*>/g, "")
      // Normalize Unicode (NFC for consistent representation)
      .normalize("NFC")
      // Remove zero-width characters
      .replace(/[\u200B-\u200D\uFEFF]/g, "")
      // Remove Hebrew cantillation marks (taamim)
      .replace(/[\u0591-\u05BD\u05BF-\u05C2\u05C4-\u05C7]/g, "")
      // Collapse multiple whitespace to single space
      .replace(/\s+/g, " ")
      // Trim
      .trim()
  );
}

/**
 * Strip HTML tags from text while preserving content
 */
export function stripHtmlTags(text: string): string {
  if (!text) return "";
  return text.replace(/<[^>]*>/g, "");
}

/**
 * Find Hebrew highlight within full text and return segmented result
 *
 * @param fullText - The complete Hebrew text (may contain HTML)
 * @param highlight - The Hebrew highlight to find
 * @returns HighlightResult with segments for rendering
 */
export function findHebrewHighlight(
  fullText: string,
  highlight: string | undefined | null
): HighlightResult {
  // Handle edge cases
  if (!fullText) {
    return {
      segments: [{ text: "", highlighted: false }],
      found: false,
    };
  }

  if (!highlight) {
    return {
      segments: [{ text: stripHtmlTags(fullText), highlighted: false }],
      found: false,
    };
  }

  // Normalize both texts for matching
  const cleanFull = stripHtmlTags(fullText);
  const normalizedFull = normalizeHebrew(fullText);
  const normalizedHighlight = normalizeHebrew(highlight);

  // Try exact substring match first
  let matchIndex = normalizedFull.indexOf(normalizedHighlight);

  if (matchIndex === -1) {
    // Try without punctuation (keep only Hebrew letters and spaces)
    const hebrewOnlyFull = normalizedFull.replace(/[^\u0590-\u05FF\s]/g, "");
    const hebrewOnlyHighlight = normalizedHighlight.replace(
      /[^\u0590-\u05FF\s]/g,
      ""
    );

    const fuzzyIndex = hebrewOnlyFull.indexOf(hebrewOnlyHighlight);

    if (fuzzyIndex === -1) {
      // No match found
      console.warn(
        `Hebrew highlight not found. Highlight: "${highlight.slice(0, 30)}..."`
      );
      return {
        segments: [{ text: cleanFull, highlighted: false }],
        found: false,
      };
    }

    // Map fuzzy index back to normalized text
    // This is approximate - find the closest position
    matchIndex = mapFuzzyIndexToOriginal(
      normalizedFull,
      hebrewOnlyFull,
      fuzzyIndex,
      hebrewOnlyHighlight.length
    );
  }

  // Create segments from the clean (non-normalized but tag-stripped) text
  // We need to map the normalized indices back to the clean text
  const mappedIndices = mapIndicesToCleanText(
    cleanFull,
    normalizedFull,
    matchIndex,
    matchIndex + normalizedHighlight.length
  );

  return createSegments(cleanFull, mappedIndices.start, mappedIndices.end);
}

/**
 * Map indices from normalized text back to clean text
 */
function mapIndicesToCleanText(
  cleanText: string,
  normalizedText: string,
  normalizedStart: number,
  normalizedEnd: number
): { start: number; end: number } {
  // If texts are same length, indices map directly
  if (cleanText.length === normalizedText.length) {
    return { start: normalizedStart, end: normalizedEnd };
  }

  // Otherwise, use ratio-based mapping (approximate)
  const ratio = cleanText.length / normalizedText.length;
  return {
    start: Math.floor(normalizedStart * ratio),
    end: Math.min(Math.ceil(normalizedEnd * ratio), cleanText.length),
  };
}

/**
 * Map fuzzy match index back to original text position
 */
function mapFuzzyIndexToOriginal(
  normalizedFull: string,
  fuzzyFull: string,
  fuzzyIndex: number,
  fuzzyLength: number
): number {
  // Use ratio-based mapping
  const ratio = normalizedFull.length / fuzzyFull.length;
  return Math.floor(fuzzyIndex * ratio);
}

/**
 * Create text segments from start/end indices
 */
function createSegments(
  text: string,
  start: number,
  end: number
): HighlightResult {
  const segments: TextSegment[] = [];

  // Clamp indices to valid range
  start = Math.max(0, Math.min(start, text.length));
  end = Math.max(start, Math.min(end, text.length));

  if (start > 0) {
    segments.push({ text: text.slice(0, start), highlighted: false });
  }

  if (end > start) {
    segments.push({ text: text.slice(start, end), highlighted: true });
  }

  if (end < text.length) {
    segments.push({ text: text.slice(end), highlighted: false });
  }

  // Handle edge case where no segments created
  if (segments.length === 0) {
    segments.push({ text: text, highlighted: false });
  }

  return {
    segments,
    found: true,
    matchStart: start,
    matchEnd: end,
  };
}

/**
 * Smart truncation that preserves the highlighted excerpt
 *
 * @param result - The highlight result
 * @param maxLength - Maximum total character length
 * @param contextChars - Characters to show before/after highlight
 * @returns Truncated segments with ellipsis markers
 */
export function smartTruncate(
  result: HighlightResult,
  maxLength: number = 300,
  contextChars: number = 50
): TextSegment[] {
  // If no highlight found, just truncate from start
  if (!result.found) {
    const fullText = result.segments[0]?.text || "";
    if (fullText.length <= maxLength) {
      return result.segments;
    }
    return [{ text: fullText.slice(0, maxLength) + "...", highlighted: false }];
  }

  // Calculate total length
  const totalLength = result.segments.reduce(
    (sum, s) => sum + s.text.length,
    0
  );

  // If already short enough, return as-is
  if (totalLength <= maxLength) {
    return result.segments;
  }

  // Find the highlighted segment
  const highlightIndex = result.segments.findIndex((s) => s.highlighted);
  const highlight = result.segments[highlightIndex];

  if (!highlight) {
    // Fallback: truncate from start
    const fullText = result.segments.map((s) => s.text).join("");
    return [{ text: fullText.slice(0, maxLength) + "...", highlighted: false }];
  }

  // If highlight itself is longer than maxLength, truncate it
  if (highlight.text.length >= maxLength - 10) {
    return [
      {
        text: highlight.text.slice(0, maxLength - 3) + "...",
        highlighted: true,
      },
    ];
  }

  const segments: TextSegment[] = [];
  const availableForContext = maxLength - highlight.text.length - 6; // Reserve for "..."
  const contextEach = Math.floor(availableForContext / 2);

  // Text before highlight
  const beforeSegments = result.segments.slice(0, highlightIndex);
  const beforeText = beforeSegments.map((s) => s.text).join("");

  if (beforeText.length > 0) {
    if (beforeText.length > contextEach) {
      // Truncate from the end of before text (closer to highlight)
      segments.push({
        text: "..." + beforeText.slice(-contextEach),
        highlighted: false,
      });
    } else {
      segments.push({ text: beforeText, highlighted: false });
    }
  }

  // The highlight itself
  segments.push(highlight);

  // Text after highlight
  const afterSegments = result.segments.slice(highlightIndex + 1);
  const afterText = afterSegments.map((s) => s.text).join("");

  if (afterText.length > 0) {
    if (afterText.length > contextEach) {
      // Truncate from the start of after text (closer to highlight)
      segments.push({
        text: afterText.slice(0, contextEach) + "...",
        highlighted: false,
      });
    } else {
      segments.push({ text: afterText, highlighted: false });
    }
  }

  return segments;
}

/**
 * Convert segments to a format suitable for React rendering
 */
export function renderSegments(segments: TextSegment[]): RenderableSegment[] {
  return segments.map((segment, index) => ({
    key: `seg-${index}-${segment.highlighted ? "hl" : "txt"}`,
    text: segment.text,
    highlighted: segment.highlighted,
  }));
}

/**
 * Get total text length from segments
 */
export function getSegmentsLength(segments: TextSegment[]): number {
  return segments.reduce((sum, s) => sum + s.text.length, 0);
}

/**
 * Check if the result would be truncated at a given max length
 */
export function wouldBeTruncated(
  result: HighlightResult,
  maxLength: number
): boolean {
  const totalLength = getSegmentsLength(result.segments);
  return totalLength > maxLength;
}

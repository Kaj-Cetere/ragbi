 # Hebrew Excerpt Highlighting Solution

## Problem Statement

When the LLM cites Torah sources, it provides:
1. **Full Hebrew text** from Sefaria (often the entire seif/commentary)
2. **Translation/explanation** (the `context` field) which may only cover part of the Hebrew

**Issues:**
- Users cannot tell which portion of the Hebrew corresponds to the translation
- The citation card may truncate Hebrew text, hiding the translated portion
- No visual connection between Hebrew source and English translation

## Solution Architecture

### Core Approach: LLM-Specified Hebrew Excerpt

The LLM will specify the exact Hebrew text it's translating via a `hebrew_excerpt` attribute in the cite tag. The frontend will then:

1. **Find the excerpt** within the full Hebrew text using substring matching
2. **Highlight** the matched portion with distinct styling
3. **Smart truncate** to ensure the highlighted portion is always visible
4. **Expand on demand** to show full context

---

## Implementation Details

### 1. Backend Changes

#### A. Updated Citation Tag Format

**New XML format:**
```xml
<cite ref="Shulchan Arukh, Orach Chayim 1:1" excerpt="יתגבר כארי לעמוד בבוקר">
One should strengthen himself like a lion to arise in the morning
</cite>
```

The `excerpt` attribute contains the specific Hebrew text being translated.

#### B. Modified System Prompt (citation_agent.py)

```python
system_prompt = """You are a Torah scholar assistant specializing in Shulchan Arukh and its commentaries.

CITATION FORMAT:
When citing a source, use this XML tag format:
<cite ref="EXACT_REFERENCE" excerpt="HEBREW_EXCERPT">ENGLISH_TRANSLATION</cite>

ATTRIBUTES:
- ref: The exact reference string from AVAILABLE REFERENCES (required)
- excerpt: The specific Hebrew phrase you are translating (required when translating only part of the source)

EXAMPLES:
1. Translating specific phrase from Shulchan Arukh:
   <cite ref="Shulchan Arukh, Orach Chayim 1:1" excerpt="יתגבר כארי לעמוד בבוקר">One should strengthen himself like a lion to arise in the morning</cite>

2. Translating full seif (omit excerpt):
   <cite ref="Shulchan Arukh, Orach Chayim 1:2">The entire text translated here</cite>

3. Mishnah Berurah with specific excerpt:
   <cite ref="Mishnah Berurah 1:1" excerpt="שלא יתבייש מפני בני אדם">so that one should not be embarrassed before people</cite>

CRITICAL RULES:
1. ALWAYS include the `excerpt` attribute when translating only PART of the source
2. The excerpt must be the EXACT Hebrew text from the source (copy it precisely)
3. The excerpt should be at least 3-4 words to ensure accurate matching
4. If translating the entire source, you may omit the excerpt attribute
5. Use EXACT reference strings from the AVAILABLE REFERENCES list

CONTENT RULES:
1. For ALL citations: Provide a clear, accurate ENGLISH TRANSLATION of the Hebrew
2. Each source should be cited at most once
3. Be RELATIVELY CONCISE - cite no more than 5 sources unless necessary
4. Write in flowing paragraphs with line breaks between ideas
5. Always cite sources when stating halachic rulings
6. If the context doesn't contain enough information, say so clearly"""
```

#### C. Updated Citation Parsing (citation_agent.py)

```python
import re

async def parse_and_emit_paragraph(
    text: str,
    source_cache: dict[str, SourceText]
) -> AsyncIterator[dict]:
    """
    Parse a paragraph for <cite> tags and emit appropriate events.
    Now supports excerpt attribute for highlighting.
    """
    # Updated regex to capture optional excerpt attribute
    cite_pattern = re.compile(
        r'<cite\s+ref="([^"]+)"(?:\s+excerpt="([^"]*)")?\s*>(.*?)</cite>',
        re.DOTALL
    )
    
    last_end = 0
    
    for match in cite_pattern.finditer(text):
        # Emit any text before this citation
        before_text = text[last_end:match.start()].strip()
        if before_text:
            yield {"type": "paragraph", "content": before_text}
        
        # Extract citation details
        ref = match.group(1)
        hebrew_excerpt = match.group(2)  # May be None if not provided
        context_text = match.group(3).strip()
        
        # Look up the source in cache
        source = source_cache.get(ref)
        
        if source:
            yield {
                "type": "citation",
                "ref": ref,
                "context": context_text,
                "hebrew": source.hebrew,
                "english": source.english,
                "book": source.book,
                "hebrew_excerpt": hebrew_excerpt  # NEW FIELD
            }
        else:
            # Try smart matching for commentary subsections
            matched_source = find_best_source_match(ref, source_cache)
            
            if matched_source:
                yield {
                    "type": "citation",
                    "ref": ref,
                    "context": context_text,
                    "hebrew": matched_source.hebrew,
                    "english": matched_source.english,
                    "book": matched_source.book,
                    "hebrew_excerpt": hebrew_excerpt  # NEW FIELD
                }
            else:
                # Fallback: emit as paragraph
                yield {"type": "paragraph", "content": f"{context_text} ({ref})"}
        
        last_end = match.end()
    
    # Emit remaining text
    remaining = text[last_end:].strip()
    if remaining:
        yield {"type": "paragraph", "content": remaining}
```

#### D. Updated Event Emission (main.py)

In the streaming handler, add the new field:

```python
elif event["type"] == "citation":
    citation_count += 1
    yield f"data: {json.dumps({
        'type': 'citation',
        'ref': event['ref'],
        'context': event['context'],
        'hebrew': event['hebrew'],
        'english': event['english'],
        'book': event.get('book', ''),
        'hebrew_excerpt': event.get('hebrew_excerpt')  # NEW FIELD
    })}\n\n"
```

---

### 2. Frontend Changes

#### A. Updated API Types (utils/api.ts)

```typescript
export interface CitationEvent {
  ref: string;
  context: string;
  hebrew: string;
  english: string;
  book?: string;
  hebrew_excerpt?: string;  // NEW: The specific Hebrew portion being translated
}

export interface StreamEvent {
  type: StreamEventType;
  message?: string;
  content?: string;
  data?: Source[];
  // Citation-specific fields
  ref?: string;
  context?: string;
  hebrew?: string;
  english?: string;
  book?: string;
  hebrew_excerpt?: string;  // NEW FIELD
  // ... rest of fields
}
```

#### B. New Utility: Hebrew Text Highlighting (utils/hebrewHighlight.ts)

```typescript
/**
 * Hebrew Text Highlighting Utilities
 * 
 * Provides functions for finding and highlighting Hebrew excerpts
 * within larger Hebrew text blocks, with smart truncation.
 */

export interface HighlightResult {
  /** The full text with highlight markers */
  segments: TextSegment[];
  /** Whether a match was found */
  found: boolean;
  /** Start index of the match in original text */
  matchStart?: number;
  /** End index of the match in original text */
  matchEnd?: number;
}

export interface TextSegment {
  text: string;
  highlighted: boolean;
}

/**
 * Normalize Hebrew text for matching
 * Removes HTML tags, extra whitespace, and normalizes Unicode
 */
export function normalizeHebrew(text: string): string {
  return text
    // Remove HTML tags
    .replace(/<[^>]*>/g, '')
    // Normalize Unicode (NFD then NFC for consistent representation)
    .normalize('NFC')
    // Remove zero-width characters
    .replace(/[\u200B-\u200D\uFEFF]/g, '')
    // Collapse multiple whitespace to single space
    .replace(/\s+/g, ' ')
    // Trim
    .trim();
}

/**
 * Find Hebrew excerpt within full text and return segmented result
 * 
 * @param fullText - The complete Hebrew text (may contain HTML)
 * @param excerpt - The Hebrew excerpt to find
 * @returns HighlightResult with segments for rendering
 */
export function findHebrewExcerpt(
  fullText: string,
  excerpt: string | undefined | null
): HighlightResult {
  if (!excerpt || !fullText) {
    return {
      segments: [{ text: fullText || '', highlighted: false }],
      found: false
    };
  }

  // Normalize both texts for matching
  const normalizedFull = normalizeHebrew(fullText);
  const normalizedExcerpt = normalizeHebrew(excerpt);

  // Find the excerpt in the full text
  const matchIndex = normalizedFull.indexOf(normalizedExcerpt);

  if (matchIndex === -1) {
    // Try fuzzy matching (remove punctuation and compare)
    const fuzzyFull = normalizedFull.replace(/[^\u0590-\u05FF\s]/g, '');
    const fuzzyExcerpt = normalizedExcerpt.replace(/[^\u0590-\u05FF\s]/g, '');
    const fuzzyIndex = fuzzyFull.indexOf(fuzzyExcerpt);

    if (fuzzyIndex === -1) {
      return {
        segments: [{ text: fullText, highlighted: false }],
        found: false
      };
    }

    // Map fuzzy index back to original - use approximate position
    const ratio = fuzzyIndex / fuzzyFull.length;
    const approxStart = Math.floor(normalizedFull.length * ratio);
    const approxEnd = Math.min(
      approxStart + normalizedExcerpt.length,
      normalizedFull.length
    );

    return createSegments(normalizedFull, approxStart, approxEnd);
  }

  return createSegments(
    normalizedFull,
    matchIndex,
    matchIndex + normalizedExcerpt.length
  );
}

function createSegments(
  text: string,
  start: number,
  end: number
): HighlightResult {
  const segments: TextSegment[] = [];

  if (start > 0) {
    segments.push({ text: text.slice(0, start), highlighted: false });
  }

  segments.push({ text: text.slice(start, end), highlighted: true });

  if (end < text.length) {
    segments.push({ text: text.slice(end), highlighted: false });
  }

  return {
    segments,
    found: true,
    matchStart: start,
    matchEnd: end
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
  if (!result.found) {
    // No highlight - just truncate from start
    const fullText = result.segments[0]?.text || '';
    if (fullText.length <= maxLength) {
      return result.segments;
    }
    return [
      { text: fullText.slice(0, maxLength) + '...', highlighted: false }
    ];
  }

  // Calculate total length
  const totalLength = result.segments.reduce((sum, s) => sum + s.text.length, 0);
  
  if (totalLength <= maxLength) {
    return result.segments;
  }

  // Find the highlighted segment
  const highlightIndex = result.segments.findIndex(s => s.highlighted);
  const highlight = result.segments[highlightIndex];
  
  if (!highlight) {
    return result.segments;
  }

  // If highlight itself is longer than maxLength, show it with ellipsis
  if (highlight.text.length >= maxLength - 10) {
    return [
      { text: highlight.text.slice(0, maxLength - 3) + '...', highlighted: true }
    ];
  }

  const segments: TextSegment[] = [];
  const remainingChars = maxLength - highlight.text.length - 6; // Reserve for "..."
  const contextEach = Math.floor(remainingChars / 2);

  // Before highlight
  const before = result.segments.slice(0, highlightIndex);
  const beforeText = before.map(s => s.text).join('');
  
  if (beforeText.length > contextEach) {
    segments.push({
      text: '...' + beforeText.slice(-contextEach),
      highlighted: false
    });
  } else if (beforeText.length > 0) {
    segments.push({ text: beforeText, highlighted: false });
  }

  // Highlight
  segments.push(highlight);

  // After highlight
  const after = result.segments.slice(highlightIndex + 1);
  const afterText = after.map(s => s.text).join('');
  
  if (afterText.length > contextEach) {
    segments.push({
      text: afterText.slice(0, contextEach) + '...',
      highlighted: false
    });
  } else if (afterText.length > 0) {
    segments.push({ text: afterText, highlighted: false });
  }

  return segments;
}

/**
 * Render segments to a React-safe structure
 * Returns an array of objects suitable for mapping in JSX
 */
export function renderSegments(
  segments: TextSegment[]
): Array<{ key: string; text: string; highlighted: boolean }> {
  return segments.map((segment, index) => ({
    key: `segment-${index}`,
    text: segment.text,
    highlighted: segment.highlighted
  }));
}
```

#### C. Updated InlineCitation Component

```tsx
"use client";

import { motion } from "framer-motion";
import { BookOpen, ChevronDown, ChevronUp } from "lucide-react";
import { useState, useMemo } from "react";
import {
  findHebrewExcerpt,
  smartTruncate,
  renderSegments,
} from "@/utils/hebrewHighlight";

interface InlineCitationProps {
  ref_: string;
  context: string;
  hebrew: string;
  english: string;
  book?: string;
  hebrew_excerpt?: string;  // NEW: The specific Hebrew portion being translated
  index: number;
  onClick?: () => void;
}

export function InlineCitation({
  ref_,
  context,
  hebrew,
  english,
  book,
  hebrew_excerpt,
  index,
  onClick,
}: InlineCitationProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [showFullHebrew, setShowFullHebrew] = useState(false);

  // Extract a short title from the ref
  const shortTitle = ref_.split(",").slice(0, 2).join(",");

  // Process Hebrew text with highlighting
  const hebrewHighlightResult = useMemo(() => {
    return findHebrewExcerpt(hebrew, hebrew_excerpt);
  }, [hebrew, hebrew_excerpt]);

  // Get truncated or full segments
  const displaySegments = useMemo(() => {
    if (showFullHebrew) {
      return renderSegments(hebrewHighlightResult.segments);
    }
    const truncated = smartTruncate(hebrewHighlightResult, 300, 60);
    return renderSegments(truncated);
  }, [hebrewHighlightResult, showFullHebrew]);

  // Check if Hebrew is truncated
  const isHebrewTruncated = useMemo(() => {
    const fullLength = hebrewHighlightResult.segments
      .reduce((sum, s) => sum + s.text.length, 0);
    return fullLength > 300;
  }, [hebrewHighlightResult]);

  // Determine if we have a valid highlight
  const hasHighlight = hebrewHighlightResult.found && hebrew_excerpt;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{
        delay: index * 0.05,
        duration: 0.4,
        ease: [0.25, 0.46, 0.45, 0.94],
      }}
      className="my-6 rounded-xl overflow-hidden"
      style={{
        backgroundColor: "var(--color-bg-surface)",
        border: "1px solid var(--color-border)",
        boxShadow: "var(--shadow-sm)",
      }}
    >
      {/* Header with reference */}
      <div
        className="px-4 py-3 flex items-center justify-between cursor-pointer transition-colors hover:opacity-90"
        style={{ backgroundColor: "var(--color-bg-surface-2)" }}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <BookOpen
            size={14}
            style={{ color: "var(--color-accent-secondary)" }}
          />
          <span
            className="text-xs font-bold uppercase tracking-wider font-sans"
            style={{ color: "var(--color-accent-secondary)" }}
          >
            {shortTitle}
          </span>
          {/* Indicator when highlight is present */}
          {hasHighlight && (
            <span
              className="text-[10px] px-1.5 py-0.5 rounded"
              style={{
                backgroundColor: "var(--color-highlight-bg)",
                color: "var(--color-accent-primary)",
              }}
            >
              excerpt
            </span>
          )}
        </div>
        <button className="p-1" style={{ color: "var(--color-text-light)" }}>
          {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
      </div>

      {/* Content section */}
      <motion.div
        initial={false}
        animate={{
          height: isExpanded ? "auto" : 0,
          opacity: isExpanded ? 1 : 0,
        }}
        transition={{ duration: 0.25, ease: "easeInOut" }}
        className="overflow-hidden"
      >
        <div className="px-4 py-4 space-y-4">
          {/* Hebrew text with highlighting */}
          <div
            className="he text-xl leading-relaxed pb-3"
            style={{
              color: "var(--color-text-main)",
              borderBottom: "1px solid var(--color-border)",
            }}
          >
            {displaySegments.map(({ key, text, highlighted }) => (
              <span
                key={key}
                className={highlighted ? "hebrew-highlight" : ""}
                style={
                  highlighted
                    ? {
                        backgroundColor: "var(--color-highlight-bg)",
                        borderRadius: "3px",
                        padding: "1px 3px",
                        boxDecorationBreak: "clone",
                        WebkitBoxDecorationBreak: "clone",
                      }
                    : undefined
                }
              >
                {text}
              </span>
            ))}
            
            {/* Show more/less button for Hebrew */}
            {isHebrewTruncated && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowFullHebrew(!showFullHebrew);
                }}
                className="inline-block mr-2 text-sm"
                style={{ color: "var(--color-accent-secondary)" }}
              >
                {showFullHebrew ? "פחות" : "עוד"}
              </button>
            )}
          </div>

          {/* Translation label when excerpt is highlighted */}
          {hasHighlight && (
            <div
              className="text-[10px] font-sans font-semibold uppercase tracking-wider"
              style={{ color: "var(--color-accent-secondary)" }}
            >
              Translation of highlighted text:
            </div>
          )}

          {/* English translation */}
          {context && (
            <div
              className="font-serif text-sm leading-relaxed"
              style={{
                color: "var(--color-text-muted)",
                fontStyle: hasHighlight ? "normal" : "italic",
              }}
            >
              {context.slice(0, 400)}
              {context.length > 400 && "..."}
            </div>
          )}

          {/* View on Sefaria link */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              onClick?.();
            }}
            className="text-xs font-sans font-medium flex items-center gap-1 transition-colors hover:opacity-80"
            style={{ color: "var(--color-accent-primary)" }}
          >
            View full source
            <span>&rarr;</span>
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}
```

#### D. Updated Page Component (page.tsx) - ResponseSegment Type

```typescript
// Type for response segments (paragraphs and inline citations)
interface ResponseSegment {
  id: string;
  type: "paragraph" | "citation";
  content?: string;
  ref?: string;
  context?: string;
  hebrew?: string;
  english?: string;
  book?: string;
  hebrew_excerpt?: string;  // NEW FIELD
}

// In the event handler:
case "citation":
  setIsLoading(false);
  if (!firstContentTime) {
    firstContentTime = now;
    // ... metrics
  }
  if (event.ref) {
    const newSegment: ResponseSegment = {
      id: `seg-${segmentCounter++}`,
      type: "citation",
      ref: event.ref,
      context: event.context,
      hebrew: event.hebrew,
      english: event.english,
      book: event.book,
      hebrew_excerpt: event.hebrew_excerpt,  // NEW FIELD
}

// In the event handler:
case "citation":
  setIsLoading(false);
  if (!firstContentTime) {
    firstContentTime = now;
    // ... metrics
  }
  if (event.ref) {
    const newSegment: ResponseSegment = {
      id: `seg-${segmentCounter++}`,
      type: "citation",
      ref: event.ref,
      context: event.context,
      hebrew: event.hebrew,
      english: event.english,
      book: event.book,
      hebrew_excerpt: event.hebrew_excerpt,  // NEW FIELD
    };
    setResponseSegments((prev) => [...prev, newSegment]);
  }
  break;
```

#### E. Updated InlineCitation Rendering (in page.tsx)

```tsx
<InlineCitation
  key={segment.id}
  ref_={segment.ref || ""}
  context={segment.context || ""}
  hebrew={segment.hebrew || ""}
  english={segment.english || ""}
  book={segment.book}
  hebrew_excerpt={segment.hebrew_excerpt}  // NEW PROP
  index={index}
  onClick={() => segment.ref && openSidebar(segment.ref)}
/>
```

#### F. CSS Additions (globals.css)

```css
/* ══════════════════════════════════════════════════════════════════════
   HEBREW HIGHLIGHT STYLES
   ══════════════════════════════════════════════════════════════════════ */

/* Highlighted Hebrew text in citations */
.hebrew-highlight {
  background-color: var(--color-highlight-bg);
  border-radius: 3px;
  padding: 1px 4px;
  box-decoration-break: clone;
  -webkit-box-decoration-break: clone;
  transition: background-color 0.2s ease;
}

.hebrew-highlight:hover {
  background-color: rgba(217, 119, 87, 0.2);
}

/* Pulse animation for newly highlighted text */
@keyframes highlightPulse {
  0% { background-color: rgba(217, 119, 87, 0.3); }
  50% { background-color: rgba(217, 119, 87, 0.15); }
  100% { background-color: rgba(217, 119, 87, 0.12); }
}

.hebrew-highlight-animate {
  animation: highlightPulse 1s ease-out;
}

/* Excerpt badge styling */
.excerpt-badge {
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 4px;
  background-color: var(--color-highlight-bg);
  color: var(--color-accent-primary);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

/* Show more/less button in Hebrew text */
.hebrew-expand-btn {
  display: inline-block;
  margin-right: 8px;
  font-size: 14px;
  color: var(--color-accent-secondary);
  cursor: pointer;
  font-family: var(--font-sans);
  transition: color 0.2s ease;
}

.hebrew-expand-btn:hover {
  color: var(--color-accent-primary);
}

/* Translation label styling */
.translation-label {
  font-size: 10px;
  font-family: var(--font-sans);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--color-accent-secondary);
  margin-bottom: 4px;
}
```

---

## Testing Plan

### 1. Unit Tests for Hebrew Matching

```typescript
// __tests__/hebrewHighlight.test.ts

import {
  normalizeHebrew,
  findHebrewExcerpt,
  smartTruncate,
} from '@/utils/hebrewHighlight';

describe('normalizeHebrew', () => {
  it('removes HTML tags', () => {
    const input = '<b>שלום</b> <i>עולם</i>';
    expect(normalizeHebrew(input)).toBe('שלום עולם');
  });

  it('normalizes whitespace', () => {
    const input = 'שלום   עולם\n\nטוב';
    expect(normalizeHebrew(input)).toBe('שלום עולם טוב');
  });
});

describe('findHebrewExcerpt', () => {
  const fullText = 'יתגבר כארי לעמוד בבוקר לעבודת בוראו';

  it('finds exact match', () => {
    const excerpt = 'יתגבר כארי';
    const result = findHebrewExcerpt(fullText, excerpt);
    
    expect(result.found).toBe(true);
    expect(result.segments.length).toBe(2);
    expect(result.segments[0].highlighted).toBe(true);
    expect(result.segments[0].text).toBe('יתגבר כארי');
  });

  it('finds match in middle', () => {
    const excerpt = 'לעמוד בבוקר';
    const result = findHebrewExcerpt(fullText, excerpt);
    
    expect(result.found).toBe(true);
    expect(result.segments.length).toBe(3);
    expect(result.segments[1].highlighted).toBe(true);
  });

  it('handles no excerpt gracefully', () => {
    const result = findHebrewExcerpt(fullText, null);
    
    expect(result.found).toBe(false);
    expect(result.segments[0].text).toBe(fullText);
  });
});

describe('smartTruncate', () => {
  it('preserves highlighted portion in truncation', () => {
    const longText = 'א'.repeat(200) + 'HIGHLIGHT' + 'ב'.repeat(200);
    const result = findHebrewExcerpt(longText, 'HIGHLIGHT');
    const truncated = smartTruncate(result, 100, 20);
    
    // The highlighted portion should be present
    const highlightedSegment = truncated.find(s => s.highlighted);
    expect(highlightedSegment).toBeDefined();
    expect(highlightedSegment?.text).toContain('HIGHLIGHT');
  });
});
```

### 2. Integration Test Scenarios

| Scenario | Hebrew Text | Excerpt | Expected Behavior |
|----------|-------------|---------|-------------------|
| Full match | "שלום עולם" | "שלום עולם" | Entire text highlighted |
| Partial start | "אבג דהו זחט" | "אבג" | First word highlighted |
| Partial middle | "אבג דהו זחט" | "דהו" | Middle word highlighted |
| Partial end | "אבג דהו זחט" | "זחט" | Last word highlighted |
| Long text truncated | 500 chars | 20 chars in middle | Shows "...context[highlight]context..." |
| No excerpt | "שלום עולם" | undefined | No highlighting, normal display |
| No match | "שלום עולם" | "אבג" | No highlighting, logs warning |

---

## Migration Strategy

### Phase 1: Backend Update (Non-Breaking)
1. Update citation_agent.py with new prompt and parsing
2. Add hebrew_excerpt to event emission
3. Deploy backend - old frontend will ignore new field

### Phase 2: Frontend Update
1. Add hebrewHighlight.ts utility
2. Update InlineCitation component
3. Update page.tsx types and rendering
4. Add CSS styles
5. Deploy frontend

### Phase 3: Monitoring
1. Monitor logs for "excerpt not found" warnings
2. Adjust fuzzy matching if needed
3. Tune LLM prompt based on excerpt quality

---

## Edge Cases Handled

1. **No excerpt provided**: Falls back to showing full Hebrew without highlights
2. **Excerpt not found**: Shows full Hebrew, logs warning for debugging
3. **Very long excerpt**: Truncates with ellipsis while keeping highlight visible
4. **Hebrew with HTML tags**: Strips tags before matching, preserves for display
5. **Unicode normalization**: Handles various Hebrew Unicode representations
6. **Empty texts**: Graceful handling of null/undefined/empty strings

---

## Benefits

1. **Clear visual connection** between Hebrew source and translation
2. **Smart truncation** ensures relevant text is always visible
3. **Expandable view** for full context when needed
4. **Non-breaking migration** - old clients work with new backend
5. **RTL-safe** highlighting that respects Hebrew text direction
6. **Accessible** - uses semantic highlighting, not just color



/**
 * Hebrew Text Highlighting Utilities
 *
 * Provides functions for finding and highlighting Hebrew excerpts
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
 * Find Hebrew excerpt within full text and return segmented result
 *
 * @param fullText - The complete Hebrew text (may contain HTML)
 * @param excerpt - The Hebrew excerpt to find
 * @returns HighlightResult with segments for rendering
 */
export function findHebrewExcerpt(
  fullText: string,
  excerpt: string | undefined | null
): HighlightResult {
  // Handle edge cases
  if (!fullText) {
    return {
      segments: [{ text: "", highlighted: false }],
      found: false,
    };
  }

  if (!excerpt) {
    return {
      segments: [{ text: stripHtmlTags(fullText), highlighted: false }],
      found: false,
    };
  }

  // Normalize both texts for matching
  const cleanFull = stripHtmlTags(fullText);
  const normalizedFull = normalizeHebrew(fullText);
  const normalizedExcerpt = normalizeHebrew(excerpt);

  // Try exact substring match first
  let matchIndex = normalizedFull.indexOf(normalizedExcerpt);

  if (matchIndex === -1) {
    // Try without punctuation (keep only Hebrew letters and spaces)
    const hebrewOnlyFull = normalizedFull.replace(/[^\u0590-\u05FF\s]/g, "");
    const hebrewOnlyExcerpt = normalizedExcerpt.replace(
      /[^\u0590-\u05FF\s]/g,
      ""
    );

    const fuzzyIndex = hebrewOnlyFull.indexOf(hebrewOnlyExcerpt);

    if (fuzzyIndex === -1) {
      // No match found
      console.warn(
        `Hebrew excerpt not found. Excerpt: "${excerpt.slice(0, 30)}..."`
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
      hebrewOnlyExcerpt.length
    );
  }

  // Create segments from the clean (non-normalized but tag-stripped) text
  // We need to map the normalized indices back to the clean text
  const mappedIndices = mapIndicesToCleanText(
    cleanFull,
    normalizedFull,
    matchIndex,
    matchIndex + normalizedExcerpt.length
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


















"use client";

import { motion } from "framer-motion";
import { BookOpen, ChevronDown, ChevronUp } from "lucide-react";
import { useState, useMemo } from "react";
import {
  findHebrewExcerpt,
  smartTruncate,
  renderSegments,
  wouldBeTruncated,
} from "@/utils/hebrewHighlight";

interface InlineCitationProps {
  ref_: string;
  context: string;
  hebrew: string;
  english: string;
  book?: string;
  hebrew_excerpt?: string; // The specific Hebrew portion being translated
  index: number;
  onClick?: () => void;
}

export function InlineCitation({
  ref_,
  context,
  hebrew,
  english,
  book,
  hebrew_excerpt,
  index,
  onClick,
}: InlineCitationProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [showFullHebrew, setShowFullHebrew] = useState(false);

  // Extract a short title from the ref
  const shortTitle = ref_.split(",").slice(0, 2).join(",");

  // Process Hebrew text with highlighting
  const hebrewHighlightResult = useMemo(() => {
    return findHebrewExcerpt(hebrew, hebrew_excerpt);
  }, [hebrew, hebrew_excerpt]);

  // Check if Hebrew would be truncated
  const isHebrewTruncated = useMemo(() => {
    return wouldBeTruncated(hebrewHighlightResult, 300);
  }, [hebrewHighlightResult]);

  // Get display segments (truncated or full)
  const displaySegments = useMemo(() => {
    if (showFullHebrew) {
      return renderSegments(hebrewHighlightResult.segments);
    }
    const truncated = smartTruncate(hebrewHighlightResult, 300, 60);
    return renderSegments(truncated);
  }, [hebrewHighlightResult, showFullHebrew]);

  // Determine if we have a valid highlight
  const hasHighlight = hebrewHighlightResult.found && !!hebrew_excerpt;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{
        delay: index * 0.05,
        duration: 0.4,
        ease: [0.25, 0.46, 0.45, 0.94],
      }}
      className="my-6 rounded-xl overflow-hidden"
      style={{
        backgroundColor: "var(--color-bg-surface)",
        border: "1px solid var(--color-border)",
        boxShadow: "var(--shadow-sm)",
      }}
    >
      {/* Header with reference */}
      <div
        className="px-4 py-3 flex items-center justify-between cursor-pointer transition-colors hover:opacity-90"
        style={{ backgroundColor: "var(--color-bg-surface-2)" }}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <BookOpen
            size={14}
            style={{ color: "var(--color-accent-secondary)" }}
          />
          <span
            className="text-xs font-bold uppercase tracking-wider font-sans"
            style={{ color: "var(--color-accent-secondary)" }}
          >
            {shortTitle}
          </span>
          {/* Badge indicating excerpt highlighting is active */}
          {hasHighlight && (
            <span
              className="text-[10px] px-1.5 py-0.5 rounded font-semibold"
              style={{
                backgroundColor: "var(--color-highlight-bg)",
                color: "var(--color-accent-primary)",
              }}
            >
              excerpt
            </span>
          )}
        </div>
        <button className="p-1" style={{ color: "var(--color-text-light)" }}>
          {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
      </div>

      {/* Content section */}
      <motion.div
        initial={false}
        animate={{
          height: isExpanded ? "auto" : 0,
          opacity: isExpanded ? 1 : 0,
        }}
        transition={{ duration: 0.25, ease: "easeInOut" }}
        className="overflow-hidden"
      >
        <div className="px-4 py-4 space-y-4">
          {/* Hebrew text with highlighting */}
          <div
            className="he text-xl leading-relaxed pb-3"
            style={{
              color: "var(--color-text-main)",
              borderBottom: "1px solid var(--color-border)",
            }}
          >
            {displaySegments.map(({ key, text, highlighted }) => (
              <span
                key={key}
                style={
                  highlighted
                    ? {
                        backgroundColor: "var(--color-highlight-bg)",
                        borderRadius: "3px",
                        padding: "2px 4px",
                        boxDecorationBreak: "clone",
                        WebkitBoxDecorationBreak: "clone",
                      }
                    : undefined
                }
              >
                {text}
              </span>
            ))}

            {/* Show more/less toggle for long Hebrew text */}
            {isHebrewTruncated && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowFullHebrew(!showFullHebrew);
                }}
                className="inline-block mr-2 text-sm font-sans"
                style={{
                  color: "var(--color-accent-secondary)",
                  marginRight: "8px",
                }}
              >
                {showFullHebrew ? "הצג פחות" : "הצג עוד"}
              </button>
            )}
          </div>

          {/* Translation label when excerpt is highlighted */}
          {hasHighlight && (
            <div
              className="text-[10px] font-sans font-semibold uppercase tracking-wider"
              style={{ color: "var(--color-accent-secondary)" }}
            >
              Translation of highlighted portion:
            </div>
          )}

          {/* English translation / context from LLM */}
          {context && (
            <div
              className="font-serif text-sm leading-relaxed"
              style={{
                color: "var(--color-text-muted)",
                fontStyle: hasHighlight ? "normal" : "italic",
              }}
            >
              {context.slice(0, 400)}
              {context.length > 400 && "..."}
            </div>
          )}

          {/* View on Sefaria link */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              onClick?.();
            }}
            className="text-xs font-sans font-medium flex items-center gap-1 transition-colors hover:opacity-80"
            style={{ color: "var(--color-accent-primary)" }}
          >
            View full source
            <span>&rarr;</span>
          </button>
        </div>
      </motion.div>
    </motion.div>
  );
}










"""
Citation Agent for Torah AI

A cleaner approach to citations that:
1. Pre-fetches seifim with Hebrew text and English translation
2. Uses XML-style <cite> tags with optional excerpt attribute for partial translations
3. Streams paragraph-by-paragraph (on line breaks) for smooth animations
4. Parses and hydrates citations with actual source text and translation
5. Supports highlighting specific Hebrew excerpts that correspond to translations
"""

import os
import re
import json
import logging
import requests
import httpx
from typing import AsyncIterator, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
CHAT_MODEL = "grok-4-1-fast-non-reasoning"


@dataclass
class SourceText:
    """Pre-fetched source text with Hebrew and English."""
    ref: str
    hebrew: str
    english: str
    book: str
    siman: Optional[int] = None
    seif: Optional[int] = None


def fetch_sefaria_text(ref: str) -> Optional[dict]:
    """Fetch text from Sefaria API with Hebrew and English translation."""
    try:
        # Clean the reference for API call
        api_ref = ref.replace(" ", "_")
        url = f"https://www.sefaria.org/api/texts/{api_ref}"

        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        # Handle various response formats
        hebrew = data.get("he", "")
        english = data.get("text", "")

        # Extract seif number from ref if it exists (e.g., "Shulchan Arukh, Orach Chayim 1:2" -> seif 2)
        seif_index = None
        if ":" in ref:
            try:
                # Get the part after the last space and extract seif number after the colon
                ref_parts = ref.rsplit(" ", 1)[-1]  # Get "1:2"
                if ":" in ref_parts:
                    seif_num = int(ref_parts.split(":")[-1])  # Get 2
                    seif_index = seif_num - 1  # Convert to 0-based index
            except (ValueError, IndexError):
                pass

        # If it's a list, extract the specific seif if we have an index, otherwise join all
        if isinstance(hebrew, list):
            if seif_index is not None and 0 <= seif_index < len(hebrew):
                hebrew = str(hebrew[seif_index]) if hebrew[seif_index] else ""
            else:
                hebrew = " ".join(str(h) for h in hebrew if h)
        if isinstance(english, list):
            if seif_index is not None and 0 <= seif_index < len(english):
                english = str(english[seif_index]) if english[seif_index] else ""
            else:
                english = " ".join(str(e) for e in english if e)

        return {
            "hebrew": hebrew,
            "english": english,
            "ref": ref,
            "title": data.get("indexTitle", ref)
        }
    except Exception as e:
        logger.warning(f"Failed to fetch from Sefaria for {ref}: {e}")
        return None


def build_source_cache(hydrated_chunks: list[dict], parallel: bool = True) -> dict[str, SourceText]:
    """
    Pre-fetch all source texts from retrieved chunks.
    Returns a dict mapping ref -> SourceText with Hebrew and English.

    Args:
        hydrated_chunks: List of chunks from retrieval
        parallel: If True, fetch Sefaria texts in parallel (much faster)
    """
    import concurrent.futures
    import time

    cache_start = time.time()
    cache: dict[str, SourceText] = {}
    refs_to_fetch: list[tuple[str, dict]] = []

    # First pass: collect refs and cache what we already have
    for chunk in hydrated_chunks:
        ref = chunk.get("ref", "")
        if not ref or ref in cache:
            continue

        refs_to_fetch.append((ref, chunk))

        # Also cache commentaries (these don't need Sefaria fetch)
        for comm in chunk.get("commentaries", []):
            commentator = comm.get('commentator', 'Commentary')
            comm_text = comm.get("text", "")
            comm_ref = comm.get("ref", "")

            if comm_ref and comm_ref not in cache:
                cache[comm_ref] = SourceText(
                    ref=comm_ref,
                    hebrew=comm_text,
                    english="",
                    book=commentator
                )

            fallback_ref = f"{ref}:{commentator}"
            if fallback_ref not in cache:
                cache[fallback_ref] = SourceText(
                    ref=comm_ref or fallback_ref,
                    hebrew=comm_text,
                    english="",
                    book=commentator
                )

    # Parallel fetch from Sefaria for main refs only
    if parallel and refs_to_fetch:
        def fetch_one(ref_chunk_tuple):
            ref, chunk = ref_chunk_tuple
            hebrew = chunk.get("content", "")
            english = ""

            sefaria_data = fetch_sefaria_text(ref)
            if sefaria_data:
                if sefaria_data.get("hebrew"):
                    hebrew = sefaria_data["hebrew"]
                english = sefaria_data.get("english", "")

            return ref, SourceText(
                ref=ref,
                hebrew=hebrew,
                english=english,
                book=chunk.get("book", ""),
                siman=chunk.get("siman"),
                seif=chunk.get("seif")
            )

        # Fetch in parallel with max 5 workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(fetch_one, refs_to_fetch))
            for ref, source in results:
                cache[ref] = source
    else:
        # Sequential fallback
        for ref, chunk in refs_to_fetch:
            hebrew = chunk.get("content", "")
            english = ""

            sefaria_data = fetch_sefaria_text(ref)
            if sefaria_data:
                if sefaria_data.get("hebrew"):
                    hebrew = sefaria_data["hebrew"]
                english = sefaria_data.get("english", "")

            cache[ref] = SourceText(
                ref=ref,
                hebrew=hebrew,
                english=english,
                book=chunk.get("book", ""),
                siman=chunk.get("siman"),
                seif=chunk.get("seif")
            )

    cache_time = (time.time() - cache_start) * 1000
    logger.info(f"Built source cache with {len(cache)} entries in {cache_time:.0f}ms (parallel={parallel})")
    return cache


def build_citation_prompt(query: str, context: str, source_refs: list[str]) -> tuple[str, str]:
    """
    Build system and user prompts that instruct the LLM to use <cite> tags.
    Now includes excerpt attribute for partial translations.
    Returns (system_prompt, user_prompt).
    """
    # Format available references for the prompt (include more to cover commentaries)
    refs_list = "\n".join(f"- {ref}" for ref in source_refs[:25])
    
    system_prompt = """You are a Torah scholar assistant specializing in Shulchan Arukh and its commentaries.

CITATION FORMAT:
When citing a source, use this XML tag format:
<cite ref="EXACT_REFERENCE" excerpt="HEBREW_EXCERPT">ENGLISH_TRANSLATION</cite>

ATTRIBUTES:
- ref: The exact reference string from AVAILABLE REFERENCES (required)
- excerpt: The specific Hebrew phrase you are translating (REQUIRED when translating only part of the source)

WHEN TO USE EXCERPT:
- ALWAYS include excerpt when you are translating only a PORTION of the source text
- The excerpt helps users see exactly which Hebrew words correspond to your translation
- Copy the Hebrew text EXACTLY as it appears in the source
- Include at least 3-4 Hebrew words for accurate matching

WHEN TO OMIT EXCERPT:
- Only omit excerpt when translating the ENTIRE source text
- If the source is short (under 10 words) and you're translating all of it

EXAMPLES:

1. Translating a specific phrase from a long Shulchan Arukh seif:
   <cite ref="Shulchan Arukh, Orach Chayim 1:1" excerpt="יתגבר כארי לעמוד בבוקר">One should strengthen himself like a lion to arise in the morning</cite>

2. Translating an entire short seif (no excerpt needed):
   <cite ref="Shulchan Arukh, Orach Chayim 1:2">The complete seif translated here</cite>

3. Mishnah Berurah - translating specific explanation:
   <cite ref="Mishnah Berurah 1:1" excerpt="שלא יתבייש מפני בני אדם המלעיגים עליו">so that one should not be embarrassed before people who mock him</cite>

4. Ba'er Hetev - highlighting key phrase:
   <cite ref="Ba'er Hetev on Shulchan Arukh, Orach Chayim 1:1" excerpt="אפילו יצרו מסיתו">even if his evil inclination persuades him</cite>

CRITICAL RULES:
1. Use EXACT reference strings from the AVAILABLE REFERENCES list
2. The excerpt must be EXACT Hebrew text from the source - copy it precisely
3. ALWAYS include excerpt when translating partial text (this is essential for user experience)
4. Provide clear, accurate ENGLISH TRANSLATIONS of the Hebrew
5. Each source should be cited at most once
6. Be RELATIVELY CONCISE - cite no more than 5 sources unless clearly necessary
7. Write in flowing paragraphs with line breaks between ideas
8. Always cite sources when stating halachic rulings
9. If the context doesn't contain enough information, say so clearly

STRUCTURE:
- Start with a brief introduction
- Weave citations naturally into your explanation
- Use line breaks (double newlines) between paragraphs for readability
- Conclude with practical guidance if applicable"""

    user_prompt = f"""Here are the relevant Torah sources for this question:

{context}

---

AVAILABLE REFERENCES (use these exact strings for citations):
{refs_list}

---

USER'S QUESTION: {query}

Please provide a helpful, accurate response with citations using the <cite ref="..." excerpt="...">translation</cite> format.
Remember: ALWAYS include the excerpt attribute when translating only part of a source."""

    return system_prompt, user_prompt


async def stream_with_citations(
    query: str,
    context: str,
    hydrated_chunks: list[dict],
    xai_api_key: str
) -> AsyncIterator[dict]:
    """
    Stream LLM response with paragraph-based buffering and citation parsing.

    Yields events:
    - {"type": "source_cache_built", "duration_ms": ...} - Timing for source cache
    - {"type": "paragraph", "content": "text..."} - A complete paragraph
    - {"type": "citation", "ref": "...", "context": "...", "hebrew": "...", "english": "...", "hebrew_excerpt": "..."}
    - {"type": "done"}
    """
    import time
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user, system

    # Step 1: Pre-fetch all source texts (now parallelized)
    cache_start = time.time()
    source_cache = build_source_cache(hydrated_chunks, parallel=True)
    cache_duration = (time.time() - cache_start) * 1000
    source_refs = list(source_cache.keys())

    # Emit source cache timing
    yield {"type": "source_cache_built", "duration_ms": round(cache_duration, 2)}

    # Step 2: Build prompts
    system_prompt, user_prompt = build_citation_prompt(query, context, source_refs)

    # Step 3: Stream from xAI
    try:
        # Create xAI client and chat
        xai_client = XAIClient(api_key=xai_api_key)
        chat = xai_client.chat.create(
            model=CHAT_MODEL,
            temperature=0.3,
            max_tokens=3000
        )

        chat.append(system(system_prompt))
        chat.append(user(user_prompt))

        # Paragraph buffer for streaming
        buffer = ""

        # Note: chat.stream() is a sync generator, use regular for loop
        for response, chunk in chat.stream():
            if not chunk.content:
                continue

            buffer += chunk.content

            # Check for paragraph breaks (double newline or single newline for simpler streaming)
            # We'll emit on single newlines for responsiveness
            while '\n' in buffer:
                # Find the line break
                idx = buffer.index('\n')
                paragraph = buffer[:idx].strip()
                buffer = buffer[idx + 1:]

                if paragraph:
                    # Parse this paragraph for citations and emit
                    async for event in parse_and_emit_paragraph(paragraph, source_cache):
                        yield event

        # Emit any remaining content in buffer
        if buffer.strip():
            async for event in parse_and_emit_paragraph(buffer.strip(), source_cache):
                yield event

        yield {"type": "done"}

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield {"type": "paragraph", "content": f"Error generating response: {e}"}
        yield {"type": "done"}


def find_best_source_match(ref: str, source_cache: dict[str, SourceText]) -> Optional[SourceText]:
    """
    Find the best matching source for a reference that isn't in the cache.
    Handles various reference formats intelligently.
    
    Priority:
    1. Look for similar Sefaria refs (e.g., "Mishnah Berurah 1:1" for "Mishnah Berurah 1:2")
    2. Match by book name for any seif in that siman
    3. Fall back to base SA seif
    """
    # Try to extract the book name and numbers from the ref
    # Common formats:
    # - "Mishnah Berurah 1:3"
    # - "Ba'er Hetev on Shulchan Arukh, Orach Chayim 1:3"
    # - "Shulchan Arukh, Orach Chayim 1:1"
    
    # Strategy 1: Find any ref from the same book in the same siman
    # Extract siman number from ref
    siman_match = re.search(r'(\d+):\d+$', ref)
    if siman_match:
        siman = siman_match.group(1)
        # Find book name (everything before the numbers)
        book_part = re.sub(r'\s*\d+:\d+$', '', ref)
        
        # Look for any cached ref from the same book and siman
        for cached_ref, cached_source in source_cache.items():
            if cached_ref.startswith(book_part) and f" {siman}:" in cached_ref:
                return cached_source
    
    # Strategy 2: Match by book title alone (first available)
    for cached_ref, cached_source in source_cache.items():
        # Check if the book name matches
        if "Mishnah Berurah" in ref and "Mishnah Berurah" in cached_ref:
            return cached_source
        if "Ba'er Hetev" in ref and "Ba'er Hetev" in cached_ref:
            return cached_source
        if "Shulchan Arukh" in ref and cached_ref.startswith("Shulchan Arukh"):
            return cached_source
    
    # Strategy 3: Try partial match
    for cached_ref, cached_source in source_cache.items():
        if ref in cached_ref or cached_ref in ref:
            return cached_source
    
    return None


async def parse_and_emit_paragraph(
    text: str,
    source_cache: dict[str, SourceText]
) -> AsyncIterator[dict]:
    """
    Parse a paragraph for <cite> tags and emit appropriate events.
    Now supports excerpt attribute for Hebrew highlighting.
    
    For text segments: yields {"type": "paragraph", "content": "..."}
    For citations: yields {"type": "citation", "ref": "...", "context": "...", "hebrew": "...", "english": "...", "hebrew_excerpt": "..."}
    """
    # Updated regex to capture optional excerpt attribute
    # Matches: <cite ref="..." excerpt="...">content</cite>
    # Or: <cite ref="...">content</cite>
    cite_pattern = re.compile(
        r'<cite\s+ref="([^"]+)"(?:\s+excerpt="([^"]*)")?\s*>(.*?)</cite>',
        re.DOTALL
    )
    
    last_end = 0
    
    for match in cite_pattern.finditer(text):
        # Emit any text before this citation
        before_text = text[last_end:match.start()].strip()
        if before_text:
            yield {"type": "paragraph", "content": before_text}
        
        # Extract citation details
        ref = match.group(1)
        hebrew_excerpt = match.group(2)  # May be None if not provided
        context_text = match.group(3).strip()
        
        # Look up the source in cache
        source = source_cache.get(ref)
        
        if source:
            yield {
                "type": "citation",
                "ref": ref,
                "context": context_text,
                "hebrew": source.hebrew,
                "english": source.english,
                "book": source.book,
                "hebrew_excerpt": hebrew_excerpt  # NEW: Pass excerpt to frontend
            }
        else:
            # Source not in cache - try smart matching for commentary subsections
            matched_source = find_best_source_match(ref, source_cache)
            
            if matched_source:
                yield {
                    "type": "citation",
                    "ref": ref,
                    "context": context_text,
                    "hebrew": matched_source.hebrew,
                    "english": matched_source.english,
                    "book": matched_source.book,
                    "hebrew_excerpt": hebrew_excerpt  # NEW: Pass excerpt to frontend
                }
            else:
                # Fallback: emit as a paragraph with the citation inline
                logger.warning(f"Citation ref not found in cache: {ref}")
                yield {"type": "paragraph", "content": f"{context_text} ({ref})"}
        
        last_end = match.end()
    
    # Emit any remaining text after the last citation
    remaining = text[last_end:].strip()
    if remaining:
        yield {"type": "paragraph", "content": remaining}


# --- MAIN EXPORT FUNCTION ---

async def generate_response_with_citations_stream(
    query: str,
    context: str,
    retrieved_chunks: list[dict],
    xai_api_key: str
) -> AsyncIterator[dict]:
    """
    Main entry point for citation-aware response generation.

    Args:
        query: User's question
        context: Formatted RAG context
        retrieved_chunks: Hydrated chunks from vector search
        xai_api_key: API key for xAI

    Yields:
        Events for frontend rendering:
        - {"type": "paragraph", "content": "..."}
        - {"type": "citation", "ref": "...", "context": "...", "hebrew": "...", "english": "...", "hebrew_excerpt": "..."}
        - {"type": "done"}
    """
    async for event in stream_with_citations(
        query=query,
        context=context,
        hydrated_chunks=retrieved_chunks,
        xai_api_key=xai_api_key
    ):
        yield event






/* ══════════════════════════════════════════════════════════════════════
   HEBREW EXCERPT HIGHLIGHTING STYLES
   
   Add these styles to your globals.css file to support the Hebrew
   excerpt highlighting feature in citation cards.
   ══════════════════════════════════════════════════════════════════════ */

/* Highlighted Hebrew text in citations */
.hebrew-highlight {
  background-color: var(--color-highlight-bg);
  border-radius: 3px;
  padding: 2px 4px;
  /* Handle text that wraps across lines */
  box-decoration-break: clone;
  -webkit-box-decoration-break: clone;
  transition: background-color 0.2s ease;
}

/* Slightly darker on hover to indicate interactivity */
.hebrew-highlight:hover {
  background-color: rgba(217, 119, 87, 0.2);
}

/* Pulse animation for newly rendered highlights */
@keyframes highlightPulse {
  0% {
    background-color: rgba(217, 119, 87, 0.35);
  }
  50% {
    background-color: rgba(217, 119, 87, 0.18);
  }
  100% {
    background-color: rgba(217, 119, 87, 0.12);
  }
}

.hebrew-highlight-animate {
  animation: highlightPulse 1s ease-out;
}

/* Excerpt badge shown in citation header */
.excerpt-badge {
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 4px;
  background-color: var(--color-highlight-bg);
  color: var(--color-accent-primary);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

/* Show more/less button for Hebrew text expansion */
.hebrew-expand-btn {
  display: inline-block;
  margin-right: 8px;
  font-size: 14px;
  color: var(--color-accent-secondary);
  cursor: pointer;
  font-family: var(--font-sans);
  transition: color 0.2s ease;
  background: none;
  border: none;
  padding: 0;
}

.hebrew-expand-btn:hover {
  color: var(--color-accent-primary);
}

/* Translation label that appears above the English when excerpt is highlighted */
.translation-label {
  font-size: 10px;
  font-family: var(--font-sans);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--color-accent-secondary);
  margin-bottom: 4px;
}

/* Enhanced citation card when excerpt highlighting is active */
.citation-card-highlighted {
  border-left-color: var(--color-accent-primary);
  border-left-width: 4px;
}

/* Smooth scroll behavior for when we auto-scroll to highlighted content */
.citation-content-scroll {
  scroll-behavior: smooth;
}

/* Connector line visual (optional - connects highlight to translation) */
.highlight-connector {
  position: relative;
}

.highlight-connector::after {
  content: '';
  position: absolute;
  left: 50%;
  bottom: -8px;
  width: 2px;
  height: 8px;
  background-color: var(--color-accent-primary);
  opacity: 0.3;
}

/* Tooltip-style indication that text is highlighted */
.highlight-tooltip {
  position: relative;
}

.highlight-tooltip::before {
  content: 'מתורגם';
  position: absolute;
  top: -20px;
  right: 0;
  font-size: 9px;
  padding: 2px 6px;
  background-color: var(--color-accent-primary);
  color: white;
  border-radius: 3px;
  opacity: 0;
  transition: opacity 0.2s ease;
  pointer-events: none;
}

.highlight-tooltip:hover::before {
  opacity: 1;
}

/* RTL-specific adjustments for the highlight */
[dir="rtl"] .hebrew-highlight,
.he .hebrew-highlight {
  /* Ensure proper padding direction for RTL text */
  padding-right: 4px;
  padding-left: 4px;
}

/* Ellipsis styling for truncated text */
.hebrew-ellipsis {
  color: var(--color-text-light);
  font-weight: normal;
}

/* Container for Hebrew text with consistent styling */
.hebrew-text-container {
  font-family: var(--font-hebrew);
  direction: rtl;
  text-align: right;
  line-height: 1.8;
}

/* Responsive adjustments */
@media (max-width: 640px) {
  .hebrew-highlight {
    padding: 1px 3px;
    border-radius: 2px;
  }
  
  .excerpt-badge {
    font-size: 9px;
    padding: 1px 4px;
  }
  
  .translation-label {
    font-size: 9px;
  }
}

/* Dark mode support (if you implement dark mode later) */
@media (prefers-color-scheme: dark) {
  .hebrew-highlight {
    background-color: rgba(217, 119, 87, 0.25);
  }
  
  .hebrew-highlight:hover {
    background-color: rgba(217, 119, 87, 0.35);
  }
}












/**
 * API utilities for connecting to the FastAPI backend
 * Updated with hebrew_excerpt support for citation highlighting
 */

// Add this at the top of the file
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Source {
  rank: number;
  ref: string;
  content: string;
  similarity: number;
  rerank_score?: number;
  book: string;
  siman?: number;
  seif?: number;
  commentaries: Array<{
    commentator: string;
    text: string;
  }>;
  hydrated: boolean;
}

export interface SefariaVerse {
  number: number;
  he: string;
  en: string;
  isTarget: boolean;
}

export interface SefariaText {
  ref: string;
  title: string;
  heTitle: string;
  verses: SefariaVerse[];
  targetVerse?: number;
}

export type StreamEventType = 'status' | 'sources' | 'chunk' | 'paragraph' | 'citation' | 'done' | 'error' | 'metrics' | 'metrics_summary';

export interface CitationEvent {
  ref: string;
  context: string;
  hebrew: string;
  english: string;
  book?: string;
  hebrew_excerpt?: string;  // NEW: The specific Hebrew portion being translated
}

// Performance metrics types
export interface MetricsDetails {
  dimensions?: number;
  chunks_found?: number;
  similarity_max?: number;
  similarity_avg?: number;
  was_reranked?: boolean;
  chunks_after?: number;
  top_rerank_score?: number;
  chunks_hydrated?: number;
  commentaries_fetched?: number;
  paragraphs?: number;
  citations?: number;
  approx_words?: number;
  words_per_second?: number;
}

export interface MetricsBreakdown {
  embedding?: number;
  vector_search?: number;
  reranking?: number;
  hydration?: number;
  source_cache?: number;  // Time to fetch translations from Sefaria
  time_to_first_token?: number;
  llm_generation?: number;
  total_pipeline?: number;
}

export interface MetricsCounts {
  embedding_dimensions?: number;
  chunks_retrieved?: number;
  chunks_after_rerank?: number;
  chunks_hydrated?: number;
  commentaries_fetched?: number;
  context_length?: number;
  paragraphs?: number;
  citations?: number;
  approx_words?: number;
}

export interface MetricsMetadata {
  similarity_max?: number;
  similarity_min?: number;
  similarity_avg?: number;
  reranking_applied?: boolean;
  rerank_score_max?: number;
  rerank_score_avg?: number;
  words_per_second?: number;
}

export interface MetricsSummary {
  total_duration_ms: number;
  breakdown: MetricsBreakdown;
  counts: MetricsCounts;
  metadata: MetricsMetadata;
}

export interface StreamEvent {
  type: StreamEventType;
  message?: string;
  content?: string;
  data?: Source[];
  // Citation-specific fields
  ref?: string;
  context?: string;
  hebrew?: string;
  english?: string;
  book?: string;
  hebrew_excerpt?: string;  // NEW: The specific Hebrew portion being translated
  // Metrics-specific fields
  stage?: string;
  duration_ms?: number;
  details?: MetricsDetails;
  // Metrics summary fields
  total_duration_ms?: number;
  breakdown?: MetricsBreakdown;
  counts?: MetricsCounts;
  metadata?: MetricsMetadata;
}

// Frontend performance tracking
export interface FrontendMetrics {
  queryStartTime: number;
  timeToFirstEvent?: number;
  timeToFirstContent?: number;
  timeToSources?: number;
  timeToComplete?: number;
  eventCount: number;
  networkLatencyEstimate?: number;
}

/**
 * Stream chat response from the API
 */
export async function* streamChat(
  query: string,
  useAgent: boolean = true
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_URL}/api/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      use_agent: useAgent,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No response body');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6));
          yield data as StreamEvent;
        } catch {
          // Ignore parse errors
        }
      }
    }
  }
}

/**
 * Fetch Sefaria text for sidebar
 */
export async function fetchSefariaText(ref: string): Promise<SefariaText> {
  const encodedRef = encodeURIComponent(ref);
  const response = await fetch(`${API_URL}/api/sefaria/text/${encodedRef}`);

  if (!response.ok) {
    throw new Error(`Failed to fetch text: ${response.status}`);
  }

  return response.json();
}











# Changes Required in page.tsx

This document outlines the specific changes needed in `frontend/app/page.tsx` to support the hebrew_excerpt feature.

## 1. Update ResponseSegment Interface

Find the ResponseSegment interface (around line 18-29) and add the hebrew_excerpt field:

```typescript
// BEFORE:
interface ResponseSegment {
  id: string;
  type: "paragraph" | "citation";
  content?: string;
  ref?: string;
  context?: string;
  hebrew?: string;
  english?: string;
  book?: string;
}

// AFTER:
interface ResponseSegment {
  id: string;
  type: "paragraph" | "citation";
  content?: string;
  ref?: string;
  context?: string;
  hebrew?: string;
  english?: string;
  book?: string;
  hebrew_excerpt?: string;  // NEW: The specific Hebrew portion being translated
}
```

## 2. Update Citation Event Handler

Find the citation case in the event handler (around line 130-145) and add hebrew_excerpt:

```typescript
// BEFORE:
case "citation":
  setIsLoading(false);
  if (!firstContentTime) {
    firstContentTime = now;
    setFrontendMetrics((prev) => ({
      ...prev,
      timeToFirstContent: firstContentTime! - queryStartTime,
    }));
  }
  if (event.ref) {
    const newSegment: ResponseSegment = {
      id: `seg-${segmentCounter++}`,
      type: "citation",
      ref: event.ref,
      context: event.context,
      hebrew: event.hebrew,
      english: event.english,
      book: event.book,
    };
    setResponseSegments((prev) => [...prev, newSegment]);
  }
  break;

// AFTER:
case "citation":
  setIsLoading(false);
  if (!firstContentTime) {
    firstContentTime = now;
    setFrontendMetrics((prev) => ({
      ...prev,
      timeToFirstContent: firstContentTime! - queryStartTime,
    }));
  }
  if (event.ref) {
    const newSegment: ResponseSegment = {
      id: `seg-${segmentCounter++}`,
      type: "citation",
      ref: event.ref,
      context: event.context,
      hebrew: event.hebrew,
      english: event.english,
      book: event.book,
      hebrew_excerpt: event.hebrew_excerpt,  // NEW: Pass excerpt for highlighting
    };
    setResponseSegments((prev) => [...prev, newSegment]);
  }
  break;
```

## 3. Update InlineCitation Rendering

Find where InlineCitation is rendered (around line 280-290) and add the hebrew_excerpt prop:

```typescript
// BEFORE:
<InlineCitation
  key={segment.id}
  ref_={segment.ref || ""}
  context={segment.context || ""}
  hebrew={segment.hebrew || ""}
  english={segment.english || ""}
  book={segment.book}
  index={index}
  onClick={() => segment.ref && openSidebar(segment.ref)}
/>

// AFTER:
<InlineCitation
  key={segment.id}
  ref_={segment.ref || ""}
  context={segment.context || ""}
  hebrew={segment.hebrew || ""}
  english={segment.english || ""}
  book={segment.book}
  hebrew_excerpt={segment.hebrew_excerpt}  // NEW: Pass excerpt for highlighting
  index={index}
  onClick={() => segment.ref && openSidebar(segment.ref)}
/>
```

## Summary of Changes

| Location | Change |
|----------|--------|
| ResponseSegment interface | Add `hebrew_excerpt?: string` |
| Citation event handler | Add `hebrew_excerpt: event.hebrew_excerpt` |
| InlineCitation render | Add `hebrew_excerpt={segment.hebrew_excerpt}` |

These are the only three changes needed in page.tsx!














# Changes needed in main.py for hebrew_excerpt support
# 
# The main.py file needs minimal changes - just passing through the new field.
# 
# Location: In the chat_stream endpoint, inside the citation event handler
# 
# Find this section (around line 290-300):

# BEFORE:
# --------
elif event["type"] == "citation":
    citation_count += 1
    yield f"data: {json.dumps({'type': 'citation', 'ref': event['ref'], 'context': event['context'], 'hebrew': event['hebrew'], 'english': event['english'], 'book': event.get('book', '')})}\n\n"

# AFTER:
# --------
elif event["type"] == "citation":
    citation_count += 1
    yield f"data: {json.dumps({
        'type': 'citation',
        'ref': event['ref'],
        'context': event['context'],
        'hebrew': event['hebrew'],
        'english': event['english'],
        'book': event.get('book', ''),
        'hebrew_excerpt': event.get('hebrew_excerpt')  # NEW: Pass hebrew excerpt for highlighting
    })}\n\n"

# That's the only change needed in main.py!
# The citation_agent.py already handles parsing the excerpt attribute
# and including it in the event dictionary.

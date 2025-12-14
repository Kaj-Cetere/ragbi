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

"use client";

import { motion } from "framer-motion";
import { BookOpen, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";

interface InlineCitationProps {
  ref_: string;
  context: string;
  hebrew: string;
  english: string;
  book?: string;
  index: number;
  onClick?: () => void;
}

export function InlineCitation({
  ref_,
  context,
  hebrew,
  english,
  book,
  index,
  onClick,
}: InlineCitationProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  // Extract a short title from the ref
  const shortTitle = ref_.split(",").slice(0, 2).join(",");

  // Strip HTML tags from Hebrew text for display
  const cleanHebrew = hebrew.replace(/<[^>]*>/g, "").slice(0, 300);
  const cleanEnglish = english.replace(/<[^>]*>/g, "").slice(0, 400);

  return (
    <motion.div
      initial={{ opacity: 0, y: 12, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ 
        delay: index * 0.05, 
        duration: 0.4,
        ease: [0.25, 0.46, 0.45, 0.94]
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
          opacity: isExpanded ? 1 : 0 
        }}
        transition={{ duration: 0.25, ease: "easeInOut" }}
        className="overflow-hidden"
      >
        <div className="px-4 py-4 space-y-4">
          {/* Context (LLM's explanation) */}
          {context && (
            <p
              className="text-sm font-sans leading-relaxed"
              style={{ color: "var(--color-text-main)" }}
            >
              {context}
            </p>
          )}

          {/* Hebrew text */}
          {cleanHebrew && (
            <div
              className="he text-xl leading-relaxed pb-3"
              style={{
                color: "var(--color-text-main)",
                borderBottom: "1px solid var(--color-border)",
              }}
            >
              {cleanHebrew}
              {hebrew.length > 300 && "..."}
            </div>
          )}

          {/* English translation */}
          {cleanEnglish && (
            <div
              className="font-serif text-sm leading-relaxed italic"
              style={{ color: "var(--color-text-muted)" }}
            >
              {cleanEnglish}
              {english.length > 400 && "..."}
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

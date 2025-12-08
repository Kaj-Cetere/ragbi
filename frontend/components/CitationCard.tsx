"use client";

import { motion } from "framer-motion";
import { BookOpen } from "lucide-react";
import type { Source } from "@/utils/api";

interface CitationCardProps {
  source: Source;
  onClick: () => void;
  index: number;
}

export function CitationCard({ source, onClick, index }: CitationCardProps) {
  // Extract a short title from the ref
  const shortTitle = source.ref.split(",").slice(0, 2).join(",");

  // Get first commentary if available
  const firstCommentary = source.commentaries?.[0];

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1, duration: 0.4 }}
      className="citation-card my-8"
      onClick={onClick}
    >
      {/* Reference label */}
      <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider font-sans mb-4" style={{ color: 'var(--color-accent-secondary)' }}>
        <BookOpen size={14} />
        <span>{shortTitle}</span>
      </div>

      {/* Hebrew text */}
      <div className="he text-2xl leading-relaxed mb-3.5" style={{ color: 'var(--color-text-main)' }}>
        {source.content.slice(0, 200)}
        {source.content.length > 200 && "..."}
      </div>

      {/* Commentary preview if available */}
      {firstCommentary && (
        <div className="font-serif text-base leading-relaxed italic" style={{ color: 'var(--color-text-muted)' }}>
          <span className="not-italic font-sans text-xs uppercase tracking-wide" style={{ color: 'var(--color-accent-secondary)' }}>
            {firstCommentary.commentator}:
          </span>{" "}
          {firstCommentary.text.slice(0, 150)}...
        </div>
      )}

      {/* Similarity score badge */}
      <div className="absolute top-4 right-4 text-xs font-mono" style={{ color: 'var(--color-text-light)' }}>
        {(source.similarity * 100).toFixed(0)}%
      </div>
    </motion.div>
  );
}

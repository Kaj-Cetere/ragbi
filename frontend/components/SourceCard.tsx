"use client";

import { motion } from "framer-motion";
import { BookOpen, ExternalLink } from "lucide-react";
import type { Source } from "@/utils/api";

interface SourceCardProps {
  source: Source;
  onClick: () => void;
  index: number;
}

export function SourceCard({ source, onClick, index }: SourceCardProps) {
  // Extract a readable title from the ref
  const shortTitle = source.ref.split(",").slice(0, 2).join(",");

  // Calculate relevance percentage (use rerank_score if available, else similarity)
  const relevance = source.rerank_score ?? source.similarity;
  const relevancePercent = (relevance * 100).toFixed(0);

  // Truncate Hebrew content for display
  const hebrewPreview = source.content.length > 250
    ? source.content.slice(0, 250) + "..."
    : source.content;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        delay: index * 0.05,
        duration: 0.35,
        ease: [0.25, 0.46, 0.45, 0.94]
      }}
      className="group relative rounded-xl p-5 mb-3 cursor-pointer transition-all duration-200"
      style={{
        backgroundColor: 'var(--color-bg-surface)',
        border: '1px solid var(--color-border)',
      }}
      onClick={onClick}
      whileHover={{
        scale: 1.01,
        boxShadow: 'var(--shadow-md)'
      }}
      whileTap={{ scale: 0.995 }}
    >
      {/* Rank badge */}
      <div
        className="absolute -top-2 -left-2 w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold"
        style={{
          backgroundColor: 'var(--color-accent-primary)',
          color: 'white'
        }}
      >
        {source.rank}
      </div>

      {/* Header row */}
      <div className="flex items-start justify-between gap-3 mb-3">
        {/* Reference title */}
        <div className="flex items-center gap-2">
          <BookOpen size={14} style={{ color: 'var(--color-accent-secondary)' }} />
          <span
            className="text-sm font-semibold font-sans"
            style={{ color: 'var(--color-text-main)' }}
          >
            {shortTitle}
          </span>
        </div>

        {/* Relevance score */}
        <div
          className="flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-mono"
          style={{
            backgroundColor: 'var(--color-bg-surface-2)',
            color: 'var(--color-text-light)'
          }}
        >
          <span>{relevancePercent}%</span>
        </div>
      </div>

      {/* Hebrew text */}
      <div
        className="he text-lg leading-relaxed mb-2"
        style={{ color: 'var(--color-text-main)' }}
      >
        {hebrewPreview}
      </div>

      {/* Book tag */}
      <div className="flex items-center justify-between">
        <span
          className="text-xs font-sans uppercase tracking-wide"
          style={{ color: 'var(--color-text-muted)' }}
        >
          {source.book}
        </span>

        {/* View link */}
        <div
          className="flex items-center gap-1 text-xs opacity-0 group-hover:opacity-100 transition-opacity"
          style={{ color: 'var(--color-accent-primary)' }}
        >
          <span>View in Sefer</span>
          <ExternalLink size={12} />
        </div>
      </div>
    </motion.div>
  );
}

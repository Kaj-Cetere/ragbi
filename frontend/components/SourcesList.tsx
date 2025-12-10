"use client";

import { motion } from "framer-motion";
import type { Source } from "@/utils/api";

interface SourcesListProps {
  sources: Source[];
  onSourceClick: (ref: string) => void;
}

export function SourcesList({ sources, onSourceClick }: SourcesListProps) {
  if (!sources.length) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="mt-12 pt-8"
      style={{ borderTop: '1px solid var(--color-border)' }}
    >
      <div className="text-xs font-bold uppercase tracking-wider font-sans mb-4" style={{ color: 'var(--color-text-light)' }}>
        Additional Sources
      </div>

      <div className="space-y-1">
        {sources.map((source) => (
          <div
            key={source.ref}
            className="source-link"
            onClick={() => onSourceClick(source.ref)}
          >
            {source.ref}
          </div>
        ))}
      </div>
    </motion.div>
  );
}

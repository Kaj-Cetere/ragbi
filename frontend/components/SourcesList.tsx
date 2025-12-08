"use client";

import { motion } from "framer-motion";
import { Scroll } from "lucide-react";
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

      <div className="space-y-2">
        {sources.map((source) => (
          <div
            key={source.ref}
            className="source-item"
            onClick={() => onSourceClick(source.ref)}
          >
            <div className="w-9 h-9 rounded-full flex items-center justify-center" style={{ backgroundColor: 'var(--color-accent-light)', color: 'var(--color-accent-secondary)' }}>
              <Scroll size={18} />
            </div>
            <div>
              <div className="font-semibold text-sm font-sans" style={{ color: 'var(--color-text-main)' }}>
                {source.ref}
              </div>
              <div className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                {source.book} • {(source.similarity * 100).toFixed(0)}% match
              </div>
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

"use client";

import { motion } from "framer-motion";
import { ArrowRight, TrendingUp, TrendingDown, Minus } from "lucide-react";
import type { RankedChunk } from "@/utils/api";

interface ChunkRankingViewProps {
  preRerank: RankedChunk[];
  postRerank: RankedChunk[];
  wasReranked: boolean;
  isVisible: boolean;
  onToggle: () => void;
}

function ChunkRow({
  chunk,
  showRerankScore,
  originalRank,
}: {
  chunk: RankedChunk;
  showRerankScore: boolean;
  originalRank?: number;
}) {
  // Calculate rank change for post-rerank items
  const rankChange = originalRank ? originalRank - chunk.rank : 0;

  return (
    <div
      className="p-3 rounded-lg mb-2"
      style={{
        backgroundColor: "var(--color-bg-surface)",
        border: "1px solid var(--color-border)",
      }}
    >
      {/* Header row with rank and ref */}
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          <span
            className="w-6 h-6 flex items-center justify-center rounded-full text-xs font-bold"
            style={{
              backgroundColor: "var(--color-accent-primary)",
              color: "white",
            }}
          >
            {chunk.rank}
          </span>
          <span
            className="text-sm font-medium truncate max-w-[180px]"
            style={{ color: "var(--color-text-main)" }}
            title={chunk.ref}
          >
            {chunk.ref.split(",").slice(0, 2).join(",")}
          </span>
        </div>

        {/* Rank change indicator for post-rerank */}
        {showRerankScore && originalRank && rankChange !== 0 && (
          <div
            className="flex items-center gap-0.5 text-xs font-medium px-1.5 py-0.5 rounded"
            style={{
              backgroundColor:
                rankChange > 0
                  ? "rgba(34, 197, 94, 0.15)"
                  : "rgba(239, 68, 68, 0.15)",
              color: rankChange > 0 ? "rgb(34, 197, 94)" : "rgb(239, 68, 68)",
            }}
          >
            {rankChange > 0 ? (
              <TrendingUp size={12} />
            ) : (
              <TrendingDown size={12} />
            )}
            <span>{Math.abs(rankChange)}</span>
          </div>
        )}
      </div>

      {/* Content preview */}
      <p
        className="text-xs leading-relaxed mb-2 he"
        style={{ color: "var(--color-text-muted)" }}
      >
        {chunk.content.length > 120
          ? chunk.content.slice(0, 120) + "..."
          : chunk.content}
      </p>

      {/* Scores row */}
      <div className="flex items-center gap-3 text-xs">
        <div className="flex items-center gap-1">
          <span style={{ color: "var(--color-text-light)" }}>Cosine:</span>
          <span
            className="font-mono font-medium"
            style={{ color: "var(--color-accent-secondary)" }}
          >
            {(chunk.similarity * 100).toFixed(1)}%
          </span>
        </div>
        {showRerankScore && chunk.rerank_score !== undefined && (
          <div className="flex items-center gap-1">
            <span style={{ color: "var(--color-text-light)" }}>Rerank:</span>
            <span
              className="font-mono font-medium"
              style={{ color: "var(--color-accent-primary)" }}
            >
              {(chunk.rerank_score * 100).toFixed(1)}%
            </span>
          </div>
        )}
        {showRerankScore && originalRank && (
          <div className="flex items-center gap-1">
            <span style={{ color: "var(--color-text-light)" }}>Was #</span>
            <span
              className="font-mono"
              style={{ color: "var(--color-text-muted)" }}
            >
              {originalRank}
            </span>
          </div>
        )}
      </div>

      {/* Book tag */}
      <div className="mt-1.5">
        <span
          className="text-[10px] uppercase tracking-wide"
          style={{ color: "var(--color-text-light)" }}
        >
          {chunk.book}
        </span>
      </div>
    </div>
  );
}

export function ChunkRankingView({
  preRerank,
  postRerank,
  wasReranked,
  isVisible,
  onToggle,
}: ChunkRankingViewProps) {
  if (!preRerank.length && !postRerank.length) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50">
      {/* Toggle button */}
      <button
        onClick={onToggle}
        className="absolute -top-10 right-0 px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
        style={{
          backgroundColor: "var(--color-bg-surface)",
          border: "1px solid var(--color-border)",
          color: "var(--color-text-muted)",
        }}
      >
        {isVisible ? "Hide" : "Show"} Ranking Comparison
      </button>

      {/* Main panel */}
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, y: 20, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 20, scale: 0.95 }}
          transition={{ duration: 0.2 }}
          className="rounded-xl overflow-hidden"
          style={{
            backgroundColor: "var(--color-bg-body)",
            border: "1px solid var(--color-border)",
            boxShadow: "var(--shadow-lg)",
            maxHeight: "70vh",
            width: "720px",
          }}
        >
          {/* Header */}
          <div
            className="px-4 py-3 flex items-center justify-between"
            style={{
              backgroundColor: "var(--color-bg-surface)",
              borderBottom: "1px solid var(--color-border)",
            }}
          >
            <h3
              className="font-semibold text-sm"
              style={{ color: "var(--color-text-main)" }}
            >
              Ranking Comparison (Top 10)
            </h3>
            <div
              className="flex items-center gap-1.5 px-2 py-1 rounded text-xs"
              style={{
                backgroundColor: wasReranked
                  ? "rgba(34, 197, 94, 0.15)"
                  : "rgba(239, 68, 68, 0.15)",
                color: wasReranked ? "rgb(34, 197, 94)" : "rgb(239, 68, 68)",
              }}
            >
              {wasReranked ? "Reranker Applied" : "Reranker Failed"}
            </div>
          </div>

          {/* Two column layout */}
          <div className="flex">
            {/* Pre-rerank column */}
            <div
              className="flex-1 p-3 overflow-y-auto"
              style={{
                maxHeight: "calc(70vh - 50px)",
                borderRight: "1px solid var(--color-border)",
              }}
            >
              <div
                className="text-xs font-medium uppercase tracking-wide mb-3 flex items-center gap-1.5"
                style={{ color: "var(--color-text-light)" }}
              >
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: "var(--color-accent-secondary)" }}
                />
                Cosine Similarity (Before)
              </div>
              {preRerank.map((chunk) => (
                <ChunkRow
                  key={`pre-${chunk.rank}-${chunk.ref}`}
                  chunk={chunk}
                  showRerankScore={false}
                />
              ))}
            </div>

            {/* Arrow separator */}
            <div
              className="flex items-center justify-center px-2"
              style={{ backgroundColor: "var(--color-bg-surface)" }}
            >
              <ArrowRight
                size={16}
                style={{ color: "var(--color-text-light)" }}
              />
            </div>

            {/* Post-rerank column */}
            <div
              className="flex-1 p-3 overflow-y-auto"
              style={{ maxHeight: "calc(70vh - 50px)" }}
            >
              <div
                className="text-xs font-medium uppercase tracking-wide mb-3 flex items-center gap-1.5"
                style={{ color: "var(--color-text-light)" }}
              >
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: "var(--color-accent-primary)" }}
                />
                Reranked (After)
              </div>
              {postRerank.map((chunk) => (
                <ChunkRow
                  key={`post-${chunk.rank}-${chunk.ref}`}
                  chunk={chunk}
                  showRerankScore={true}
                  originalRank={chunk.original_rank}
                />
              ))}
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}

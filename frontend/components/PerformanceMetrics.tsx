"use client";

import { useState, useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  Clock,
  Database,
  Zap,
  Server,
  Globe,
  ChevronDown,
  ChevronUp,
  Cpu,
  BarChart3,
  Timer,
  X,
  Maximize2,
  Minimize2,
} from "lucide-react";
import type {
  MetricsBreakdown,
  MetricsCounts,
  MetricsMetadata,
  FrontendMetrics,
} from "@/utils/api";

// Stage metrics for real-time tracking
export interface StageMetric {
  stage: string;
  duration_ms: number;
  details?: Record<string, unknown>;
  timestamp: number;
}

interface PerformanceMetricsProps {
  // Backend metrics
  stageMetrics: StageMetric[];
  summary: {
    total_duration_ms: number;
    breakdown: MetricsBreakdown;
    counts: MetricsCounts;
    metadata: MetricsMetadata;
  } | null;
  // Frontend metrics
  frontendMetrics: FrontendMetrics;
  // State
  isStreaming: boolean;
  isComplete: boolean;
  // Visibility control
  isVisible: boolean;
  onToggleVisibility: () => void;
}

// Format milliseconds to human readable
function formatMs(ms: number | undefined): string {
  if (ms === undefined || ms === null) return "-";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

// Format number with commas
function formatNumber(n: number | undefined): string {
  if (n === undefined || n === null) return "-";
  return n.toLocaleString();
}

// Stage display info
const STAGE_INFO: Record<
  string,
  { label: string; icon: React.ElementType; color: string }
> = {
  embedding: {
    label: "Embedding",
    icon: Cpu,
    color: "var(--color-accent-primary)",
  },
  vector_search: {
    label: "Vector Search",
    icon: Database,
    color: "#6366f1",
  },
  reranking: { label: "Reranking", icon: BarChart3, color: "#8b5cf6" },
  hydration: { label: "Hydration", icon: Globe, color: "#14b8a6" },
  source_cache: { label: "Source Cache", icon: Globe, color: "#06b6d4" },
  llm_first_token: { label: "Time to First Token", icon: Zap, color: "#f59e0b" },
  llm_complete: { label: "LLM Generation", icon: Activity, color: "#22c55e" },
};

// Pipeline stage for waterfall visualization
interface PipelineStage {
  name: string;
  duration: number;
  startOffset: number;
  color: string;
  icon: React.ElementType;
}

export function PerformanceMetrics({
  stageMetrics,
  summary,
  frontendMetrics,
  isStreaming,
  isComplete,
  isVisible,
  onToggleVisibility,
}: PerformanceMetricsProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [activeTab, setActiveTab] = useState<"overview" | "backend" | "frontend" | "details">("overview");

  // Calculate live elapsed time
  const [elapsedTime, setElapsedTime] = useState(0);

  useEffect(() => {
    if (isStreaming && frontendMetrics.queryStartTime) {
      const interval = setInterval(() => {
        setElapsedTime(Date.now() - frontendMetrics.queryStartTime);
      }, 100);
      return () => clearInterval(interval);
    } else if (isComplete && frontendMetrics.timeToComplete) {
      setElapsedTime(frontendMetrics.timeToComplete);
    }
  }, [isStreaming, isComplete, frontendMetrics]);

  // Build pipeline stages for waterfall
  const pipelineStages = useMemo((): PipelineStage[] => {
    if (!summary?.breakdown) return [];

    const stages: PipelineStage[] = [];
    let offset = 0;

    const addStage = (key: keyof MetricsBreakdown, name: string, color: string, icon: React.ElementType) => {
      const duration = summary.breakdown[key];
      if (duration !== undefined && duration > 0) {
        stages.push({ name, duration, startOffset: offset, color, icon });
        offset += duration;
      }
    };

    addStage("embedding", "Embedding", "var(--color-accent-primary)", Cpu);
    addStage("vector_search", "Vector Search", "#6366f1", Database);
    addStage("reranking", "Reranking", "#8b5cf6", BarChart3);
    addStage("hydration", "Hydration", "#14b8a6", Globe);
    addStage("source_cache", "Source Cache", "#06b6d4", Globe);
    addStage("llm_generation", "LLM", "#22c55e", Activity);

    return stages;
  }, [summary]);

  // Calculate total backend time
  const totalBackendTime = summary?.total_duration_ms || 0;

  // Toggle button when collapsed
  if (!isVisible) {
    return (
      <motion.button
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        onClick={onToggleVisibility}
        className="metrics-toggle-btn"
        title="Show Performance Metrics"
      >
        <Activity size={16} />
        {isStreaming && (
          <span className="metrics-live-dot" />
        )}
      </motion.button>
    );
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 20, scale: 0.95 }}
        transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
        className={`metrics-panel ${isFullscreen ? "metrics-panel-fullscreen" : ""}`}
      >
        {/* Header */}
        <div className="metrics-header">
          <div className="metrics-header-left">
            <Activity size={16} className="metrics-icon" />
            <span className="metrics-title">Performance Metrics</span>
            {isStreaming && (
              <span className="metrics-live-badge">
                <span className="metrics-live-dot-inline" />
                LIVE
              </span>
            )}
          </div>
          <div className="metrics-header-right">
            {/* Live timer */}
            <div className="metrics-timer">
              <Timer size={14} />
              <span>{formatMs(elapsedTime)}</span>
            </div>
            {/* Controls */}
            <button
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="metrics-control-btn"
              title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
            >
              {isFullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
            </button>
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="metrics-control-btn"
              title={isExpanded ? "Collapse" : "Expand"}
            >
              {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
            <button
              onClick={onToggleVisibility}
              className="metrics-control-btn"
              title="Close"
            >
              <X size={14} />
            </button>
          </div>
        </div>

        {/* Content */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="metrics-content"
            >
              {/* Tabs */}
              <div className="metrics-tabs">
                {(["overview", "backend", "frontend", "details"] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`metrics-tab ${activeTab === tab ? "active" : ""}`}
                  >
                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </button>
                ))}
              </div>

              {/* Tab Content */}
              <div className="metrics-tab-content">
                {activeTab === "overview" && (
                  <OverviewTab
                    pipelineStages={pipelineStages}
                    totalBackendTime={totalBackendTime}
                    frontendMetrics={frontendMetrics}
                    summary={summary}
                    isComplete={isComplete}
                    stageMetrics={stageMetrics}
                  />
                )}

                {activeTab === "backend" && (
                  <BackendTab
                    stageMetrics={stageMetrics}
                    summary={summary}
                    pipelineStages={pipelineStages}
                    totalBackendTime={totalBackendTime}
                  />
                )}

                {activeTab === "frontend" && (
                  <FrontendTab
                    frontendMetrics={frontendMetrics}
                    isComplete={isComplete}
                  />
                )}

                {activeTab === "details" && (
                  <DetailsTab summary={summary} stageMetrics={stageMetrics} />
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </AnimatePresence>
  );
}

// Overview Tab - Summary with waterfall
function OverviewTab({
  pipelineStages,
  totalBackendTime,
  frontendMetrics,
  summary,
  isComplete,
  stageMetrics,
}: {
  pipelineStages: PipelineStage[];
  totalBackendTime: number;
  frontendMetrics: FrontendMetrics;
  summary: PerformanceMetricsProps["summary"];
  isComplete: boolean;
  stageMetrics: StageMetric[];
}) {
  return (
    <div className="metrics-overview">
      {/* Key Metrics Row */}
      <div className="metrics-key-row">
        <div className="metric-card">
          <div className="metric-card-icon" style={{ backgroundColor: "rgba(217, 119, 87, 0.1)" }}>
            <Server size={16} style={{ color: "var(--color-accent-primary)" }} />
          </div>
          <div className="metric-card-content">
            <div className="metric-card-value">{formatMs(totalBackendTime)}</div>
            <div className="metric-card-label">Backend Total</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-card-icon" style={{ backgroundColor: "rgba(34, 197, 94, 0.1)" }}>
            <Zap size={16} style={{ color: "#22c55e" }} />
          </div>
          <div className="metric-card-content">
            <div className="metric-card-value">
              {formatMs(summary?.breakdown?.time_to_first_token)}
            </div>
            <div className="metric-card-label">Time to First Token</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-card-icon" style={{ backgroundColor: "rgba(99, 102, 241, 0.1)" }}>
            <Globe size={16} style={{ color: "#6366f1" }} />
          </div>
          <div className="metric-card-content">
            <div className="metric-card-value">
              {formatMs(frontendMetrics.timeToFirstEvent)}
            </div>
            <div className="metric-card-label">Network RTT</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-card-icon" style={{ backgroundColor: "rgba(245, 158, 11, 0.1)" }}>
            <Activity size={16} style={{ color: "#f59e0b" }} />
          </div>
          <div className="metric-card-content">
            <div className="metric-card-value">
              {summary?.metadata?.words_per_second
                ? `${summary.metadata.words_per_second.toFixed(1)}`
                : "-"}
            </div>
            <div className="metric-card-label">Words/sec</div>
          </div>
        </div>
      </div>

      {/* Waterfall Visualization */}
      {pipelineStages.length > 0 && (
        <div className="metrics-waterfall">
          <div className="waterfall-header">
            <span>Pipeline Waterfall</span>
            <span className="waterfall-total">{formatMs(totalBackendTime)}</span>
          </div>
          <div className="waterfall-chart">
            {pipelineStages.map((stage, idx) => {
              const widthPercent = (stage.duration / totalBackendTime) * 100;
              const leftPercent = (stage.startOffset / totalBackendTime) * 100;
              const Icon = stage.icon;

              return (
                <div key={stage.name} className="waterfall-row">
                  <div className="waterfall-label">
                    <Icon size={12} style={{ color: stage.color }} />
                    <span>{stage.name}</span>
                  </div>
                  <div className="waterfall-bar-container">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${widthPercent}%` }}
                      transition={{ duration: 0.5, delay: idx * 0.1 }}
                      className="waterfall-bar"
                      style={{
                        left: `${leftPercent}%`,
                        backgroundColor: stage.color,
                      }}
                    />
                  </div>
                  <div className="waterfall-duration">{formatMs(stage.duration)}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Live Stage Progress */}
      {!isComplete && stageMetrics.length > 0 && (
        <div className="metrics-live-stages">
          <div className="live-stages-header">Live Progress</div>
          {stageMetrics.slice(-3).map((metric, idx) => {
            const info = STAGE_INFO[metric.stage] || {
              label: metric.stage,
              icon: Activity,
              color: "var(--color-text-muted)",
            };
            const Icon = info.icon;

            return (
              <motion.div
                key={`${metric.stage}-${metric.timestamp}`}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className="live-stage-item"
              >
                <Icon size={14} style={{ color: info.color }} />
                <span className="live-stage-label">{info.label}</span>
                <span className="live-stage-duration">{formatMs(metric.duration_ms)}</span>
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// Backend Tab - Detailed backend metrics
function BackendTab({
  stageMetrics,
  summary,
  pipelineStages,
  totalBackendTime,
}: {
  stageMetrics: StageMetric[];
  summary: PerformanceMetricsProps["summary"];
  pipelineStages: PipelineStage[];
  totalBackendTime: number;
}) {
  return (
    <div className="metrics-backend">
      {/* Stage Timeline */}
      <div className="backend-section">
        <h4>Stage Breakdown</h4>
        <div className="stage-list">
          {pipelineStages.map((stage) => {
            const Icon = stage.icon;
            const percent = ((stage.duration / totalBackendTime) * 100).toFixed(1);

            return (
              <div key={stage.name} className="stage-item">
                <div className="stage-item-header">
                  <div className="stage-item-left">
                    <Icon size={14} style={{ color: stage.color }} />
                    <span>{stage.name}</span>
                  </div>
                  <div className="stage-item-right">
                    <span className="stage-percent">{percent}%</span>
                    <span className="stage-duration">{formatMs(stage.duration)}</span>
                  </div>
                </div>
                <div className="stage-bar-bg">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${percent}%` }}
                    className="stage-bar"
                    style={{ backgroundColor: stage.color }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Stage Details */}
      {stageMetrics.length > 0 && (
        <div className="backend-section">
          <h4>Stage Details</h4>
          <div className="stage-details-list">
            {stageMetrics.map((metric, idx) => {
              const info = STAGE_INFO[metric.stage];
              if (!info) return null;
              const Icon = info.icon;

              return (
                <div key={`${metric.stage}-${idx}`} className="stage-detail-item">
                  <div className="stage-detail-header">
                    <Icon size={14} style={{ color: info.color }} />
                    <span>{info.label}</span>
                    <span className="stage-detail-duration">{formatMs(metric.duration_ms)}</span>
                  </div>
                  {metric.details && Object.keys(metric.details).length > 0 && (
                    <div className="stage-detail-meta">
                      {Object.entries(metric.details).map(([key, value]) => (
                        <span key={key} className="detail-chip">
                          {key.replace(/_/g, " ")}: {typeof value === "number" ? formatNumber(value) : String(value)}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Retrieval Stats */}
      {summary?.counts && (
        <div className="backend-section">
          <h4>Retrieval Statistics</h4>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-label">Chunks Retrieved</span>
              <span className="stat-value">{formatNumber(summary.counts.chunks_retrieved)}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">After Rerank</span>
              <span className="stat-value">{formatNumber(summary.counts.chunks_after_rerank)}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Chunks Hydrated</span>
              <span className="stat-value">{formatNumber(summary.counts.chunks_hydrated)}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Commentaries</span>
              <span className="stat-value">{formatNumber(summary.counts.commentaries_fetched)}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Context Length</span>
              <span className="stat-value">{formatNumber(summary.counts.context_length)} chars</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Similarity (max)</span>
              <span className="stat-value">
                {summary.metadata?.similarity_max?.toFixed(4) || "-"}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Frontend Tab - Frontend performance metrics
function FrontendTab({
  frontendMetrics,
  isComplete,
}: {
  frontendMetrics: FrontendMetrics;
  isComplete: boolean;
}) {
  return (
    <div className="metrics-frontend">
      {/* Timing Metrics */}
      <div className="frontend-section">
        <h4>Timing Metrics</h4>
        <div className="timing-list">
          <div className="timing-item">
            <div className="timing-label">
              <Clock size={14} />
              <span>Time to First Event</span>
            </div>
            <div className="timing-value">{formatMs(frontendMetrics.timeToFirstEvent)}</div>
          </div>
          <div className="timing-item">
            <div className="timing-label">
              <Zap size={14} />
              <span>Time to First Content</span>
            </div>
            <div className="timing-value">{formatMs(frontendMetrics.timeToFirstContent)}</div>
          </div>
          <div className="timing-item">
            <div className="timing-label">
              <Database size={14} />
              <span>Time to Sources</span>
            </div>
            <div className="timing-value">{formatMs(frontendMetrics.timeToSources)}</div>
          </div>
          <div className="timing-item">
            <div className="timing-label">
              <Activity size={14} />
              <span>Time to Complete</span>
            </div>
            <div className="timing-value">
              {isComplete ? formatMs(frontendMetrics.timeToComplete) : "..."}
            </div>
          </div>
        </div>
      </div>

      {/* Stream Stats */}
      <div className="frontend-section">
        <h4>Stream Statistics</h4>
        <div className="stats-grid">
          <div className="stat-item">
            <span className="stat-label">Total Events</span>
            <span className="stat-value">{frontendMetrics.eventCount}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Est. Network Latency</span>
            <span className="stat-value">{formatMs(frontendMetrics.networkLatencyEstimate)}</span>
          </div>
        </div>
      </div>

      {/* Performance Tips */}
      <div className="frontend-section">
        <h4>Performance Analysis</h4>
        <div className="perf-tips">
          {frontendMetrics.timeToFirstEvent && frontendMetrics.timeToFirstEvent > 1000 && (
            <div className="perf-tip warning">
              High network latency detected ({formatMs(frontendMetrics.timeToFirstEvent)}).
              This may be due to geographic distance to the server.
            </div>
          )}
          {frontendMetrics.timeToFirstContent && frontendMetrics.timeToFirstContent > 3000 && (
            <div className="perf-tip warning">
              Slow time to first content. Consider optimizing embedding or search operations.
            </div>
          )}
          {!frontendMetrics.timeToFirstEvent && (
            <div className="perf-tip info">
              Waiting for data... Performance analysis will appear after the query completes.
            </div>
          )}
          {isComplete && frontendMetrics.timeToComplete && frontendMetrics.timeToComplete < 3000 && (
            <div className="perf-tip success">
              Excellent performance! Query completed in under 3 seconds.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Details Tab - Raw metrics data
function DetailsTab({
  summary,
  stageMetrics,
}: {
  summary: PerformanceMetricsProps["summary"];
  stageMetrics: StageMetric[];
}) {
  return (
    <div className="metrics-details">
      {/* Raw Summary */}
      {summary && (
        <div className="details-section">
          <h4>Summary Data</h4>
          <pre className="details-json">
            {JSON.stringify(summary, null, 2)}
          </pre>
        </div>
      )}

      {/* Stage Events */}
      <div className="details-section">
        <h4>Stage Events ({stageMetrics.length})</h4>
        <pre className="details-json">
          {JSON.stringify(stageMetrics, null, 2)}
        </pre>
      </div>
    </div>
  );
}

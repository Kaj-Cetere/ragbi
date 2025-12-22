"use client";

import { useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, ArrowRight, Search, Send, Brain, Zap } from "lucide-react";
import ReactMarkdown from "react-markdown";

import { Loader } from "@/components/Loader";
import { CitationCard } from "@/components/CitationCard";
import { SourceCard } from "@/components/SourceCard";
import { InlineCitation } from "@/components/InlineCitation";
import { SourcesList } from "@/components/SourcesList";
import { Sidebar } from "@/components/Sidebar";
import { PerformanceMetrics, type StageMetric } from "@/components/PerformanceMetrics";
import {
  streamChat,
  type Source,
  type StreamEvent,
  type MetricsBreakdown,
  type MetricsCounts,
  type MetricsMetadata,
  type FrontendMetrics,
} from "@/utils/api";

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
  hebrew_highlight?: string;  // NEW: The specific Hebrew portion being translated
}

// Suggestion pills for quick searches
const SUGGESTIONS = [
  { emoji: "🧀", text: "Dairy on Shavuot" },
  { emoji: "🕯️", text: "Shabbat Candles" },
  { emoji: "🙏", text: "Modeh Ani" },
  { emoji: "🍷", text: "Four Cups Pesach" },
];

// Metrics summary type
interface MetricsSummary {
  total_duration_ms: number;
  breakdown: MetricsBreakdown;
  counts: MetricsCounts;
  metadata: MetricsMetadata;
}

export default function Home() {
  // View state
  const [isSearching, setIsSearching] = useState(false);
  const [currentQuery, setCurrentQuery] = useState("");

  // Response mode toggle: false = AI Response (smart), true = Sources Only (fast)
  const [sourcesOnlyMode, setSourcesOnlyMode] = useState(false);

  // Input state
  const [inputValue, setInputValue] = useState("");
  const [followUpValue, setFollowUpValue] = useState("");

  // Response state
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState("Analyzing Sources...");
  const [responseSegments, setResponseSegments] = useState<ResponseSegment[]>([]);
  const [fallbackText, setFallbackText] = useState(""); // For legacy chunk-based streaming
  const [sources, setSources] = useState<Source[]>([]);
  const [sourcesOnlyResults, setSourcesOnlyResults] = useState<Source[]>([]); // For sources-only mode
  const [isComplete, setIsComplete] = useState(false);

  // Sidebar state
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedRef, setSelectedRef] = useState<string | null>(null);

  // Performance metrics state - hidden by default, unlocked with secret code
  const [metricsUnlocked, setMetricsUnlocked] = useState(false);
  const [metricsVisible, setMetricsVisible] = useState(true);
  const [stageMetrics, setStageMetrics] = useState<StageMetric[]>([]);
  const [metricsSummary, setMetricsSummary] = useState<MetricsSummary | null>(null);
  const [frontendMetrics, setFrontendMetrics] = useState<FrontendMetrics>({
    queryStartTime: 0,
    eventCount: 0,
  });

  // Refs
  const scrollerRef = useRef<HTMLDivElement>(null);

  // Secret code to unlock metrics: "1226"
  const SECRET_CODE = "1226";

  // Handle search submission
  const handleSearch = useCallback(async (query: string) => {
    if (!query.trim()) return;

    // Check for secret code to unlock metrics
    if (query.trim() === SECRET_CODE) {
      setMetricsUnlocked(true);
      setMetricsVisible(true);
      setInputValue("");
      return;
    }

    // Initialize frontend metrics tracking
    const queryStartTime = Date.now();
    let firstEventTime: number | undefined;
    let firstContentTime: number | undefined;
    let sourcesTime: number | undefined;
    let eventCount = 0;

    // Reset metrics state
    setStageMetrics([]);
    setMetricsSummary(null);
    setFrontendMetrics({
      queryStartTime,
      eventCount: 0,
    });

    // Transition to results view
    setIsSearching(true);
    setCurrentQuery(query);
    setIsLoading(true);
    setIsComplete(false);
    setResponseSegments([]);
    setFallbackText("");
    setSources([]);
    setSourcesOnlyResults([]);
    setLoadingStatus(sourcesOnlyMode ? "Finding Sources..." : "Connecting to Library...");

    let segmentCounter = 0;

    try {
      for await (const event of streamChat(query, !sourcesOnlyMode, sourcesOnlyMode)) {
        eventCount++;
        const now = Date.now();

        // Track first event time
        if (!firstEventTime) {
          firstEventTime = now;
        }

        // Update frontend metrics on each event
        setFrontendMetrics((prev) => ({
          ...prev,
          eventCount,
          timeToFirstEvent: firstEventTime ? firstEventTime - queryStartTime : undefined,
          networkLatencyEstimate: firstEventTime ? firstEventTime - queryStartTime : undefined,
        }));

        switch (event.type) {
          case "status":
            setLoadingStatus(event.message || "Processing...");
            break;

          case "sources":
            if (event.data) {
              setSources(event.data);
              if (!sourcesTime) {
                sourcesTime = now;
                setFrontendMetrics((prev) => ({
                  ...prev,
                  timeToSources: sourcesTime! - queryStartTime,
                }));
              }
            }
            break;

          case "sources_only":
            setIsLoading(false);
            if (event.data) {
              setSourcesOnlyResults(event.data);
              if (!firstContentTime) {
                firstContentTime = now;
                setFrontendMetrics((prev) => ({
                  ...prev,
                  timeToFirstContent: firstContentTime! - queryStartTime,
                }));
              }
              if (!sourcesTime) {
                sourcesTime = now;
                setFrontendMetrics((prev) => ({
                  ...prev,
                  timeToSources: sourcesTime! - queryStartTime,
                }));
              }
            }
            break;

          case "paragraph":
            setIsLoading(false);
            if (!firstContentTime) {
              firstContentTime = now;
              setFrontendMetrics((prev) => ({
                ...prev,
                timeToFirstContent: firstContentTime! - queryStartTime,
              }));
            }
            if (event.content) {
              const newSegment: ResponseSegment = {
                id: `seg-${segmentCounter++}`,
                type: "paragraph",
                content: event.content,
              };
              setResponseSegments((prev) => [...prev, newSegment]);
            }
            break;

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
                hebrew_highlight: event.hebrew_highlight,  // NEW: Pass highlight for highlighting
              };
              setResponseSegments((prev) => [...prev, newSegment]);
            }
            break;

          case "chunk":
            // Fallback for legacy streaming
            setIsLoading(false);
            if (!firstContentTime) {
              firstContentTime = now;
              setFrontendMetrics((prev) => ({
                ...prev,
                timeToFirstContent: firstContentTime! - queryStartTime,
              }));
            }
            if (event.content) {
              setFallbackText((prev) => prev + event.content);
            }
            break;

          case "metrics":
            // Track individual stage metrics from backend
            if (event.stage && event.duration_ms !== undefined) {
              const stageMetric: StageMetric = {
                stage: event.stage,
                duration_ms: event.duration_ms,
                details: event.details as Record<string, unknown>,
                timestamp: now,
              };
              setStageMetrics((prev) => [...prev, stageMetric]);
            }
            break;

          case "metrics_summary":
            // Store complete metrics summary from backend
            if (event.total_duration_ms !== undefined && event.breakdown && event.counts && event.metadata) {
              setMetricsSummary({
                total_duration_ms: event.total_duration_ms,
                breakdown: event.breakdown,
                counts: event.counts,
                metadata: event.metadata,
              });
            }
            break;

          case "done":
            setIsComplete(true);
            // Final frontend metrics update
            const completeTime = Date.now();
            setFrontendMetrics((prev) => ({
              ...prev,
              timeToComplete: completeTime - queryStartTime,
              eventCount,
            }));
            break;

          case "error":
            setIsLoading(false);
            setFallbackText(`Error: ${event.message}`);
            setIsComplete(true);
            // Update metrics on error too
            const errorTime = Date.now();
            setFrontendMetrics((prev) => ({
              ...prev,
              timeToComplete: errorTime - queryStartTime,
              eventCount,
            }));
            break;
        }
      }
    } catch (error) {
      setIsLoading(false);
      setFallbackText(
        `Error: ${error instanceof Error ? error.message : "Unknown error"}`
      );
      setIsComplete(true);
      // Update metrics on catch
      const catchTime = Date.now();
      setFrontendMetrics((prev) => ({
        ...prev,
        timeToComplete: catchTime - queryStartTime,
        eventCount,
      }));
    }
  }, [sourcesOnlyMode]);

  // Handle quick search from pills
  const handleQuickSearch = (text: string) => {
    setInputValue(text);
    handleSearch(text);
  };

  // Handle follow-up question
  const handleFollowUp = () => {
    if (followUpValue.trim()) {
      handleSearch(followUpValue);
      setFollowUpValue("");
    }
  };

  // Open sidebar with source reference
  const openSidebar = (ref: string) => {
    setSelectedRef(ref);
    setSidebarOpen(true);
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* LANDING VIEW */}
      <AnimatePresence>
        {!isSearching && (
          <motion.div
            initial={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -40 }}
            transition={{ duration: 0.5, ease: [0.8, 0, 0.2, 1] }}
            className="absolute inset-0 flex flex-col items-center justify-center z-20"
          >
            {/* Hero Logo - HALACHIC with AI emphasis */}
            <motion.h1
              className="text-3xl md:text-4xl mb-10 text-center tracking-widest select-none italic"
              style={{ fontFamily: '"Cormorant Garamond", Georgia, serif', fontWeight: 300 }}
            >
              {/* H */}
              <motion.span
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
                className="inline-block"
                style={{ color: 'var(--color-text-main)' }}
              >
                H
              </motion.span>
              {/* A */}
              <motion.span
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
                className="inline-block"
                style={{ color: 'var(--color-text-main)' }}
              >
                A
              </motion.span>
              {/* L */}
              <motion.span
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
                className="inline-block"
                style={{ color: 'var(--color-text-main)' }}
              >
                L
              </motion.span>
              {/* A - AI letter with special styling */}
              <motion.span
                initial={{ opacity: 0, scale: 0.5, filter: 'blur(10px)' }}
                animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
                transition={{ duration: 0.8, delay: 0.7, ease: [0.22, 1, 0.36, 1] }}
                className="inline-block ai-letter"
              >
                A
              </motion.span>
              {/* C */}
              <motion.span
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
                className="inline-block"
                style={{ color: 'var(--color-text-main)' }}
              >
                C
              </motion.span>
              {/* H */}
              <motion.span
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.35, ease: [0.22, 1, 0.36, 1] }}
                className="inline-block"
                style={{ color: 'var(--color-text-main)' }}
              >
                H
              </motion.span>
              {/* I - AI letter with special styling */}
              <motion.span
                initial={{ opacity: 0, scale: 0.5, filter: 'blur(10px)' }}
                animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
                transition={{ duration: 0.8, delay: 0.85, ease: [0.22, 1, 0.36, 1] }}
                className="inline-block ai-letter"
              >
                I
              </motion.span>
              {/* C */}
              <motion.span
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.45, ease: [0.22, 1, 0.36, 1] }}
                className="inline-block"
                style={{ color: 'var(--color-text-main)' }}
              >
                C
              </motion.span>
            </motion.h1>

            {/* Search Container */}
            <div className="w-full max-w-[680px] px-6">
              <div className="search-box">
                <Sparkles style={{ color: 'var(--color-accent-secondary)' }} size={20} />
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSearch(inputValue)}
                  placeholder="What do you want to learn today?"
                  className="flex-1 bg-transparent border-none text-lg outline-none font-sans font-medium"
                  style={{ color: 'var(--color-text-main)' }}
                />
                <button
                  onClick={() => handleSearch(inputValue)}
                  className="p-2.5 rounded-lg flex items-center justify-center transition-colors"
                  style={{ backgroundColor: 'var(--color-bg-surface-2)' }}
                >
                  <ArrowRight size={18} style={{ color: 'var(--color-text-main)' }} />
                </button>
              </div>

              {/* Response Mode Toggle */}
              <div className="flex justify-center mt-5">
                <div
                  className="inline-flex rounded-full p-1"
                  style={{ backgroundColor: 'var(--color-bg-surface)', border: '1px solid var(--color-border)' }}
                >
                  <button
                    onClick={() => setSourcesOnlyMode(false)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all duration-200 ${
                      !sourcesOnlyMode ? 'shadow-sm' : ''
                    }`}
                    style={{
                      backgroundColor: !sourcesOnlyMode ? 'var(--color-accent-primary)' : 'transparent',
                      color: !sourcesOnlyMode ? 'white' : 'var(--color-text-muted)',
                    }}
                  >
                    <Brain size={16} />
                    <span>AI Response</span>
                  </button>
                  <button
                    onClick={() => setSourcesOnlyMode(true)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all duration-200 ${
                      sourcesOnlyMode ? 'shadow-sm' : ''
                    }`}
                    style={{
                      backgroundColor: sourcesOnlyMode ? 'var(--color-accent-secondary)' : 'transparent',
                      color: sourcesOnlyMode ? 'white' : 'var(--color-text-muted)',
                    }}
                  >
                    <Zap size={16} />
                    <span>Sources Only</span>
                  </button>
                </div>
              </div>

              {/* Suggestions */}
              <div className="flex gap-3 justify-center mt-6 flex-wrap">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s.text}
                    onClick={() => handleQuickSearch(s.text)}
                    className="pill"
                  >
                    {s.text}
                  </button>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* RESULTS VIEW */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: isSearching ? 1 : 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
        className={`flex-1 flex flex-col overflow-hidden ${
          isSearching ? "" : "pointer-events-none"
        }`}
      >
        {/* Stream Content */}
        <div
          ref={scrollerRef}
          className="flex-1 overflow-y-auto px-6 py-10 pb-48"
        >
          <div className="max-w-[700px] mx-auto">
            {/* Top Bar (Question + Mode Indicator) */}
            <div className="px-6 py-5 mb-6 -mx-6 border-b glass-bg" style={{ borderColor: 'var(--color-border)' }}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Search size={18} style={{ color: 'var(--color-text-light)' }} />
                  <span className="font-serif text-xl font-semibold" style={{ color: 'var(--color-text-main)' }}>
                    {currentQuery}
                  </span>
                </div>
                {/* Mode badge */}
                <div
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium"
                  style={{
                    backgroundColor: sourcesOnlyMode ? 'var(--color-accent-secondary)' : 'var(--color-accent-primary)',
                    color: 'white',
                    opacity: 0.9
                  }}
                >
                  {sourcesOnlyMode ? <Zap size={12} /> : <Brain size={12} />}
                  <span>{sourcesOnlyMode ? 'Sources Only' : 'AI Response'}</span>
                </div>
              </div>
            </div>
            {/* Loader */}
            {isLoading && <Loader text={loadingStatus} />}

            {/* Sources Only Mode - Display all reranked sources */}
            {sourcesOnlyMode && sourcesOnlyResults.length > 0 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
                className="space-y-1"
              >
                {sourcesOnlyResults.map((source, i) => (
                  <SourceCard
                    key={source.ref}
                    source={source}
                    index={i}
                    onClick={() => openSidebar(source.ref)}
                  />
                ))}
              </motion.div>
            )}

            {/* AI Response Mode - Response Segments (paragraphs and inline citations) */}
            {!sourcesOnlyMode && responseSegments.map((segment, index) => (
              segment.type === "paragraph" ? (
                <motion.div
                  key={segment.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
                  className="prose-kesher mb-4"
                >
                  <ReactMarkdown>{segment.content || ""}</ReactMarkdown>
                </motion.div>
              ) : (
                <InlineCitation
                  key={segment.id}
                  ref_={segment.ref || ""}
                  context={segment.context || ""}
                  hebrew={segment.hebrew || ""}
                  english={segment.english || ""}
                  book={segment.book}
                  hebrew_highlight={segment.hebrew_highlight}  // NEW: Pass highlight for highlighting
                  index={index}
                  onClick={() => segment.ref && openSidebar(segment.ref)}
                />
              )
            ))}

            {/* Fallback for legacy chunk-based streaming */}
            {!sourcesOnlyMode && fallbackText && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="prose-kesher mb-8"
              >
                <ReactMarkdown>{fallbackText}</ReactMarkdown>
              </motion.div>
            )}

            {/* Citation Cards for top sources - only show when stream is complete and NO inline citations exist (legacy fallback) */}
            {!sourcesOnlyMode && isComplete && responseSegments.filter(s => s.type === "citation").length === 0 && sources.slice(0, 3).map((source, i) => (
              <CitationCard
                key={source.ref}
                source={source}
                index={i}
                onClick={() => openSidebar(source.ref)}
              />
            ))}

            {/* Additional Sources List - only show when stream is complete */}
            {!sourcesOnlyMode && isComplete && sources.length > 3 && (
              <SourcesList
                sources={sources.slice(3)}
                onSourceClick={openSidebar}
              />
            )}
          </div>
        </div>

        {/* Follow-up Input Dock */}
        <div className="absolute bottom-0 left-0 right-0 px-6 py-6" style={{ background: 'linear-gradient(to top, var(--color-bg-body) 70%, transparent)' }}>
          <div className="max-w-[700px] mx-auto">
            <div className="rounded-2xl px-5 py-3.5 flex items-center gap-3 transition-all" style={{ backgroundColor: 'var(--color-bg-surface)', border: '1px solid var(--color-border)', boxShadow: 'var(--shadow-md)' }}>
              <input
                type="text"
                value={followUpValue}
                onChange={(e) => setFollowUpValue(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleFollowUp()}
                placeholder="ask another question"
                className="flex-1 bg-transparent border-none text-sm outline-none font-sans"
                style={{ color: 'var(--color-text-main)' }}
              />
              <button
                onClick={handleFollowUp}
                className="transition-colors"
                style={{ color: 'var(--color-accent-primary)' }}
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        </div>
      </motion.div>

      {/* SIDEBAR */}
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        selectedRef={selectedRef}
      />

      {/* PERFORMANCE METRICS PANEL - Only shown when unlocked with secret code */}
      {metricsUnlocked && isSearching && (
        <PerformanceMetrics
          stageMetrics={stageMetrics}
          summary={metricsSummary}
          frontendMetrics={frontendMetrics}
          isStreaming={isLoading}
          isComplete={isComplete}
          isVisible={metricsVisible}
          onToggleVisibility={() => setMetricsVisible(!metricsVisible)}
        />
      )}
    </div>
  );
}

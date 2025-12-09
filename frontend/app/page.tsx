"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, ArrowRight, Search, Send } from "lucide-react";
import ReactMarkdown from "react-markdown";

import { Loader } from "@/components/Loader";
import { CitationCard } from "@/components/CitationCard";
import { InlineCitation } from "@/components/InlineCitation";
import { SourcesList } from "@/components/SourcesList";
import { Sidebar } from "@/components/Sidebar";
import { streamChat, type Source, type StreamEvent } from "@/utils/api";

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
}

// Suggestion pills for quick searches
const SUGGESTIONS = [
  { emoji: "🧀", text: "Dairy on Shavuot" },
  { emoji: "🕯️", text: "Shabbat Candles" },
  { emoji: "🙏", text: "Modeh Ani" },
  { emoji: "🍷", text: "Four Cups Pesach" },
];

export default function Home() {
  // View state
  const [isSearching, setIsSearching] = useState(false);
  const [currentQuery, setCurrentQuery] = useState("");

  // Input state
  const [inputValue, setInputValue] = useState("");
  const [followUpValue, setFollowUpValue] = useState("");

  // Response state
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState("Analyzing Sources...");
  const [responseSegments, setResponseSegments] = useState<ResponseSegment[]>([]);
  const [fallbackText, setFallbackText] = useState(""); // For legacy chunk-based streaming
  const [sources, setSources] = useState<Source[]>([]);
  const [isComplete, setIsComplete] = useState(false);

  // Sidebar state
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedRef, setSelectedRef] = useState<string | null>(null);

  // Refs
  const scrollerRef = useRef<HTMLDivElement>(null);

  // Handle search submission
  const handleSearch = useCallback(async (query: string) => {
    if (!query.trim()) return;

    // Transition to results view
    setIsSearching(true);
    setCurrentQuery(query);
    setIsLoading(true);
    setIsComplete(false);
    setResponseSegments([]);
    setFallbackText("");
    setSources([]);
    setLoadingStatus("Connecting to Library...");
    
    let segmentCounter = 0;

    try {
      for await (const event of streamChat(query, true)) {
        switch (event.type) {
          case "status":
            setLoadingStatus(event.message || "Processing...");
            break;

          case "sources":
            if (event.data) {
              setSources(event.data);
            }
            break;

          case "paragraph":
            setIsLoading(false);
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

          case "chunk":
            // Fallback for legacy streaming
            setIsLoading(false);
            if (event.content) {
              setFallbackText((prev) => prev + event.content);
            }
            break;

          case "done":
            setIsComplete(true);
            break;

          case "error":
            setIsLoading(false);
            setFallbackText(`Error: ${event.message}`);
            setIsComplete(true);
            break;
        }
      }
    } catch (error) {
      setIsLoading(false);
      setFallbackText(
        `Error: ${error instanceof Error ? error.message : "Unknown error"}`
      );
      setIsComplete(true);
    }
  }, []);

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

              {/* Suggestions */}
              <div className="flex gap-3 justify-center mt-8 flex-wrap">
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
        {/* Top Bar */}
        <div className="px-6 py-5 border-b glass-bg sticky top-0 z-10" style={{ borderColor: 'var(--color-border)' }}>
          <div className="max-w-[700px] mx-auto flex items-center gap-3">
            <Search size={18} style={{ color: 'var(--color-text-light)' }} />
                          <span className="font-serif text-xl font-semibold" style={{ color: 'var(--color-text-main)' }}>
              {currentQuery}
            </span>
          </div>
        </div>

        {/* Stream Content */}
        <div
          ref={scrollerRef}
          className="flex-1 overflow-y-auto px-6 py-10 pb-36"
        >
          <div className="max-w-[700px] mx-auto">
            {/* Loader */}
            {isLoading && <Loader text={loadingStatus} />}

            {/* Response Segments (paragraphs and inline citations) */}
            {responseSegments.map((segment, index) => (
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
                  index={index}
                  onClick={() => segment.ref && openSidebar(segment.ref)}
                />
              )
            ))}

            {/* Fallback for legacy chunk-based streaming */}
            {fallbackText && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="prose-kesher mb-8"
              >
                <ReactMarkdown>{fallbackText}</ReactMarkdown>
              </motion.div>
            )}

            {/* Citation Cards for top sources - only show when NO inline citations exist (legacy fallback) */}
            {!isLoading && responseSegments.filter(s => s.type === "citation").length === 0 && sources.slice(0, 3).map((source, i) => (
              <CitationCard
                key={source.ref}
                source={source}
                index={i}
                onClick={() => openSidebar(source.ref)}
              />
            ))}

            {/* Additional Sources List */}
            {!isLoading && sources.length > 3 && (
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
                placeholder="Ask a follow up question..."
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
    </div>
  );
}

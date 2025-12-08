"use client";

import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";
import { fetchSefariaText, type SefariaText } from "@/utils/api";

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  selectedRef: string | null;
}

export function Sidebar({ isOpen, onClose, selectedRef }: SidebarProps) {
  const [loading, setLoading] = useState(false);
  const [textData, setTextData] = useState<SefariaText | null>(null);
  const [error, setError] = useState<string | null>(null);
  const targetRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!selectedRef || !isOpen) return;

    async function loadText() {
      setLoading(true);
      setError(null);

      try {
        const data = await fetchSefariaText(selectedRef!);
        setTextData(data);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load text");
      } finally {
        setLoading(false);
      }
    }

    loadText();
  }, [selectedRef, isOpen]);

  // Scroll to target verse when loaded
  useEffect(() => {
    if (textData && targetRef.current) {
      setTimeout(() => {
        targetRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "center",
        });
      }, 300);
    }
  }, [textData]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 z-40"
            onClick={onClose}
          />

          {/* Sidebar */}
          <motion.aside
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="fixed right-0 top-0 bottom-0 w-[550px] max-w-full z-50 flex flex-col"
            style={{ backgroundColor: 'var(--color-bg-surface)', borderLeft: '1px solid var(--color-border)', boxShadow: "-10px 0 40px rgba(0,0,0,0.05)" }}
          >
            {/* Header */}
            <div className="h-[70px] flex items-center justify-between px-6" style={{ borderBottom: '1px solid var(--color-border)', backgroundColor: 'rgba(255, 255, 255, 0.95)', backdropFilter: 'blur(10px)' }}>
              <div>
                <div className="text-[10px] font-bold uppercase tracking-wider" style={{ color: 'var(--color-accent-secondary)' }}>
                  Source Context
                </div>
                <div className="font-bold text-base font-sans" style={{ color: 'var(--color-text-main)' }}>
                  {textData?.title || selectedRef || "Loading..."}
                </div>
              </div>
              <button
                onClick={onClose}
                className="w-8 h-8 rounded-full flex items-center justify-center transition-colors"
                style={{ backgroundColor: 'var(--color-bg-surface-2)', color: 'var(--color-text-muted)' }}
              >
                <X size={18} />
              </button>
            </div>

            {/* Body */}
            <div className="flex-1 overflow-y-auto p-8">
              {loading && (
                <div className="text-center mt-10" style={{ color: 'var(--color-text-light)' }}>
                  Fetching Chapter...
                </div>
              )}

              {error && (
                <div className="text-center mt-10" style={{ color: 'var(--color-accent-primary)' }}>
                  {error}
                </div>
              )}

              {textData && !loading && (
                <div className="space-y-2">
                  {textData.verses.map((verse) => (
                    <div
                      key={verse.number}
                      ref={verse.isTarget ? targetRef : undefined}
                      className={`context-verse ${verse.isTarget ? "target" : ""}`}
                    >
                      <div className="text-[10px] mb-1 font-bold font-sans" style={{ color: 'var(--color-text-light)' }}>
                        {verse.number} {verse.isTarget && "• QUOTED"}
                      </div>
                      <div
                        className="he text-[22px] leading-relaxed mb-1.5"
                        style={{ color: 'var(--color-text-main)' }}
                        dangerouslySetInnerHTML={{ __html: verse.he }}
                      />
                      <div
                        className="font-serif text-base leading-relaxed"
                        style={{ color: 'var(--color-text-muted)' }}
                        dangerouslySetInnerHTML={{ __html: verse.en }}
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}

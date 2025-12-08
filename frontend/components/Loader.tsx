"use client";

import { motion } from "framer-motion";

interface LoaderProps {
  text?: string;
}

export function Loader({ text = "Analyzing Sources..." }: LoaderProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 mb-5">
      <div className="relative w-16 h-16">
        {/* Center breathing orb */}
        <motion.div
          className="absolute top-1/2 left-1/2 w-4 h-4 rounded-full z-10"
          style={{
            x: "-50%",
            y: "-50%",
            backgroundColor: 'var(--color-accent-primary)',
          }}
          animate={{
            scale: [1, 1.3, 1],
            opacity: [0.8, 1, 0.8],
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        {/* Orbit 1 */}
        <motion.div
          className="absolute top-1/2 left-1/2 w-10 h-10 rounded-full"
          style={{ border: '2px solid var(--color-accent-secondary)', opacity: 0.5, x: "-50%", y: "-50%" }}
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        >
          <div
            className="absolute -top-0.5 left-1/2 w-1.5 h-1.5 rounded-full"
            style={{ backgroundColor: 'var(--color-accent-primary)' }}
          />
        </motion.div>

        {/* Orbit 2 */}
        <motion.div
          className="absolute top-1/2 left-1/2 w-16 h-16 rounded-full"
          style={{ border: '2px solid var(--color-accent-primary)', opacity: 0.3, x: "-50%", y: "-50%" }}
          animate={{ rotate: -360 }}
          transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
        >
          <div
            className="absolute -top-0.5 left-1/2 w-1.5 h-1.5 rounded-full"
            style={{ backgroundColor: 'var(--color-accent-primary)' }}
          />
        </motion.div>
      </div>

      <motion.p
        className="text-xs font-semibold uppercase tracking-widest font-sans mt-8"
        style={{ color: 'var(--color-text-light)' }}
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 1.5, repeat: Infinity }}
      >
        {text}
      </motion.p>
    </div>
  );
}

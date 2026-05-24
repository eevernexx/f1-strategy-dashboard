"use client";

import { motion } from "framer-motion";

export function TopBar({
  years,
  year,
  onYearChange,
}: {
  years: number[];
  year: number;
  onYearChange: (y: number) => void;
}) {
  return (
    <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
      <motion.div
        initial={{ opacity: 0, x: -12 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5 }}
        className="flex items-center gap-3"
      >
        <h1 className="text-xl font-extrabold tracking-tight text-ink md:text-[22px]">
          F1 Intelligence
        </h1>
        <span className="hidden h-5 w-px bg-line sm:block" />
        <span className="hidden text-[15px] font-medium text-ink-dim sm:block">
          Season Dashboard
        </span>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, x: 12 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5 }}
        className="flex items-center gap-2.5"
      >
        {/* season selector */}
        <div className="flex items-center rounded-pill border border-line bg-white/[0.03] p-1">
          {years.map((y) => (
            <button
              key={y}
              onClick={() => onYearChange(y)}
              className="relative rounded-pill px-3 py-1 text-[13px] font-semibold transition-colors"
            >
              {y === year && (
                <motion.span
                  layoutId="year-pill"
                  className="absolute inset-0 rounded-pill bg-f1"
                  transition={{ type: "spring", stiffness: 380, damping: 30 }}
                />
              )}
              <span className={`relative z-10 ${y === year ? "text-white" : "text-ink-dim"}`}>
                {y}
              </span>
            </button>
          ))}
        </div>

        <button
          aria-label="Search"
          className="grid h-9 w-9 place-items-center rounded-full border border-line bg-white/[0.03] text-ink-dim transition-colors hover:text-ink"
        >
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none">
            <circle cx="11" cy="11" r="6.5" stroke="currentColor" strokeWidth="1.8" />
            <path d="m20 20-3.2-3.2" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </button>
        <button
          aria-label="Notifications"
          className="relative grid h-9 w-9 place-items-center rounded-full border border-line bg-white/[0.03] text-ink-dim transition-colors hover:text-ink"
        >
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none">
            <path d="M6 9a6 6 0 1 1 12 0c0 5 2 6 2 6H4s2-1 2-6Z" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
            <path d="M10 19a2 2 0 0 0 4 0" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
          </svg>
          <span className="absolute right-2 top-2 h-2 w-2 rounded-full bg-f1 ring-2 ring-base" />
        </button>
      </motion.div>
    </header>
  );
}

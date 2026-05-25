"use client";

import { motion } from "framer-motion";
import { useTheme } from "@/lib/useTheme";

export function TopBar({
  years,
  year,
  onYearChange,
  viewLabel,
  onMenu,
}: {
  years: number[];
  year: number;
  onYearChange: (y: number) => void;
  viewLabel: string;
  onMenu: () => void;
}) {
  const { theme, toggle } = useTheme();

  return (
    <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
      <motion.div
        initial={{ opacity: 0, x: -12 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5 }}
        className="flex items-center gap-3"
      >
        {/* mobile menu button */}
        <button
          onClick={onMenu}
          aria-label="Open navigation"
          className="grid h-9 w-9 place-items-center rounded-xl border border-line bg-ov/[0.03] text-ink-dim transition-colors hover:text-ink md:hidden"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
            <path d="M4 7h16M4 12h16M4 17h16" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </button>

        <h1 className="text-lg font-extrabold tracking-tight text-ink md:text-[22px]">
          F1 Dashboard Visualization
        </h1>
        <span className="hidden h-5 w-px bg-line sm:block" />
        <span className="hidden text-[15px] font-medium text-ink-dim sm:block">
          {viewLabel}
        </span>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, x: 12 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5 }}
        className="flex items-center gap-2.5"
      >
        {/* advanced visualization (Streamlit app) */}
        <a
          href="https://f1-strategy-dashboard-pnq5i4cjdnqx4cqv7f53dv.streamlit.app/"
          target="_blank"
          rel="noopener noreferrer"
          title="Open the advanced Streamlit dashboard"
          className="hidden items-center gap-2 rounded-pill border border-line bg-ov/[0.03] px-3.5 py-1.5 text-[13px] font-semibold text-ink-dim transition-colors hover:border-f1 hover:text-ink sm:flex"
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
            <path d="M3 13.5 12 9l9 4.5M3 13.5 12 18l9-4.5M3 13.5V18l9 4.5 9-4.5v-4.5" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
          </svg>
          Advanced Visualization
        </a>

        {/* season selector */}
        <div className="flex items-center rounded-pill border border-line bg-ov/[0.03] p-1">
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

        {/* theme toggle */}
        <button
          onClick={toggle}
          aria-label="Toggle theme"
          title={theme === "dark" ? "Switch to light" : "Switch to dark"}
          className="grid h-9 w-9 place-items-center rounded-full border border-line bg-ov/[0.03] text-ink-dim transition-colors hover:text-ink"
        >
          {theme === "dark" ? (
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none">
              <path d="M21 12.8A8.5 8.5 0 1 1 11.2 3a6.5 6.5 0 0 0 9.8 9.8Z" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" />
            </svg>
          ) : (
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="4" stroke="currentColor" strokeWidth="1.7" />
              <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
            </svg>
          )}
        </button>
      </motion.div>
    </header>
  );
}

"use client";

import { motion } from "framer-motion";
import { ReactNode } from "react";

const ARROW = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
    <path
      d="M7 17L17 7M17 7H9M17 7v8"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

export const cardVariants = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0 },
};

export function Card({
  title,
  action,
  children,
  className = "",
  bodyClassName = "",
  showArrow = false,
  span = "",
}: {
  title?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  showArrow?: boolean;
  span?: string;
}) {
  return (
    <motion.section
      variants={cardVariants}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ y: -3 }}
      className={`card group p-5 flex flex-col ${span} ${className}`}
    >
      {(title || action || showArrow) && (
        <header className="flex items-center justify-between mb-4">
          <div className="card-title">{title}</div>
          <div className="flex items-center gap-2 text-ink-faint">
            {action}
            {showArrow && (
              <button
                aria-label="open"
                className="rounded-lg p-1 transition-colors hover:text-ink hover:bg-white/[0.06]"
              >
                {ARROW}
              </button>
            )}
          </div>
        </header>
      )}
      <div className={`flex-1 min-h-0 ${bodyClassName}`}>{children}</div>
    </motion.section>
  );
}

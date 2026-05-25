"use client";

import { motion } from "framer-motion";
import { ReactNode } from "react";

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
  span = "",
}: {
  title?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
  span?: string;
}) {
  return (
    <motion.section
      variants={cardVariants}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ y: -3 }}
      className={`card group p-5 flex flex-col ${span} ${className}`}
    >
      {(title || action) && (
        <header className="flex items-center justify-between mb-4">
          <div className="card-title">{title}</div>
          <div className="flex items-center gap-2 text-ink-faint">{action}</div>
        </header>
      )}
      <div className={`flex-1 min-h-0 ${bodyClassName}`}>{children}</div>
    </motion.section>
  );
}

"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/Card";
import type { Season } from "@/lib/types";

export function ContendersCard({ season }: { season: Season }) {
  const top = season.drivers.slice(0, 6);

  return (
    <Card title="Title contenders" showArrow span="lg:col-span-6">
      <div className="flex items-center gap-3">
        <div className="flex -space-x-3">
          {top.map((d, i) => (
            <motion.div
              key={d.code}
              initial={{ opacity: 0, scale: 0.5, x: -8 }}
              animate={{ opacity: 1, scale: 1, x: 0 }}
              transition={{ delay: 0.2 + i * 0.07, type: "spring", stiffness: 300 }}
              whileHover={{ y: -4, zIndex: 10 }}
              title={`${d.name} · ${d.points} pts`}
              className="grid h-11 w-11 place-items-center rounded-full text-[11px] font-bold text-white ring-2 ring-card"
              style={{ background: d.color }}
            >
              {d.code}
            </motion.div>
          ))}
        </div>
        <motion.button
          whileHover={{ scale: 1.08, rotate: 90 }}
          className="grid h-11 w-11 shrink-0 place-items-center rounded-full border border-dashed border-white/20 text-ink-dim hover:text-ink"
          aria-label="more"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </motion.button>
      </div>
      <p className="mt-3 text-[12px] text-ink-faint">
        Top {top.length} of {season.totals.drivers} drivers by championship points —{" "}
        <span className="text-ink-dim">{season.year} season</span>.
      </p>
    </Card>
  );
}

"use client";

import { motion } from "framer-motion";
import { roundTopScorer } from "@/lib/season";
import type { Season } from "@/lib/types";

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.03, delayChildren: 0.05 } },
};
const item = {
  hidden: { opacity: 0, y: 14 },
  show: { opacity: 1, y: 0 },
};

export function CalendarView({ season }: { season: Season }) {
  const done = season.totals.completedRounds;

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
    >
      {season.rounds.map((r, i) => {
        const completed = r.round <= done;
        const winner = completed ? roundTopScorer(season, i) : null;
        return (
          <motion.div
            key={r.round}
            variants={item}
            whileHover={{ y: -3 }}
            className="card p-4"
            style={{ borderLeft: `3px solid ${winner ? winner.color : "var(--c-line)"}` }}
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-[12px] font-bold text-ink-faint">R{r.round}</span>
              <span
                className={`pill ${completed ? "bg-pos/15 text-pos" : "bg-ov/[0.06] text-ink-faint"}`}
              >
                {completed ? "Done" : "Upcoming"}
              </span>
            </div>
            <div className="mt-2 text-[15px] font-bold leading-tight text-ink">
              {r.name.replace(" Grand Prix", "")}
            </div>
            <div className="mt-0.5 truncate text-[11px] text-ink-faint">{r.circuit}</div>

            <div className="mt-3 flex h-6 items-center">
              {winner ? (
                <div className="flex items-center gap-2">
                  <span
                    className="grid h-6 w-8 place-items-center rounded-md text-[10px] font-bold text-white"
                    style={{ background: winner.color }}
                  >
                    {winner.code}
                  </span>
                  <span className="text-[11px] text-ink-dim">Top scorer</span>
                </div>
              ) : (
                <span className="text-[11px] text-ink-faint">—</span>
              )}
            </div>
          </motion.div>
        );
      })}
    </motion.div>
  );
}

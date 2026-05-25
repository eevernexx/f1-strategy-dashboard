"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";
import { Card } from "@/components/ui/Card";
import { fmtSigned } from "@/lib/format";
import type { Season, RecentResult } from "@/lib/types";

const FILTERS = ["All", "Wins", "Podiums"] as const;
type Filter = (typeof FILTERS)[number];

function match(r: RecentResult, f: Filter) {
  if (f === "All") return true;
  if (f === "Wins") return r.result === "Win";
  return r.result === "Win" || r.result === "Podium";
}

export function RecentResultsCard({ season }: { season: Season }) {
  const [filter, setFilter] = useState<Filter>("All");
  const rows = season.recent.filter((r) => match(r, filter)).slice(0, 8);

  return (
    <Card title="Recent results" span="lg:col-span-4 lg:row-span-2">
      <div className="mb-3 flex items-center gap-1.5">
        {FILTERS.map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`relative rounded-pill px-3 py-1 text-[12px] font-semibold transition-colors ${
              filter === f ? "text-white" : "text-ink-dim"
            }`}
          >
            {filter === f && (
              <motion.span
                layoutId="filter-pill"
                className="absolute inset-0 rounded-pill bg-ov/[0.08]"
              />
            )}
            <span className="relative z-10">{f}</span>
          </button>
        ))}
      </div>

      <div className="flex flex-col">
        <AnimatePresence mode="popLayout">
          {rows.map((r, i) => (
            <motion.div
              layout
              key={`${r.round}-${r.code}`}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              transition={{ delay: i * 0.04 }}
              className="flex items-center gap-3 border-b border-line/60 py-2.5 last:border-0"
            >
              <div
                className="grid h-9 w-9 shrink-0 place-items-center rounded-full text-[11px] font-bold text-white"
                style={{ background: `${r.color}` }}
              >
                {r.code}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="truncate text-[13px] font-semibold text-ink">{r.name}</span>
                  <span
                    className={`pill ${
                      r.status === "Success"
                        ? "bg-pos/15 text-pos"
                        : "bg-neg/15 text-neg"
                    }`}
                  >
                    {r.result}
                  </span>
                </div>
                <div className="truncate text-[11px] text-ink-faint">
                  R{r.round} · {r.gp.replace(" Grand Prix", " GP")}
                </div>
              </div>
              <div
                className={`text-[13px] font-bold ${
                  r.points > 0 ? "text-ink" : "text-ink-faint"
                }`}
              >
                {fmtSigned(r.points)}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </Card>
  );
}

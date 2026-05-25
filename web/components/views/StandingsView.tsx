"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/Card";
import type { Season } from "@/lib/types";

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.03, delayChildren: 0.05 } },
};
const row = {
  hidden: { opacity: 0, x: -10 },
  show: { opacity: 1, x: 0 },
};

export function StandingsView({ season }: { season: Season }) {
  const maxD = Math.max(...season.drivers.map((d) => d.points), 1);
  const maxC = Math.max(...season.constructors.map((c) => c.points), 1);

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
      <Card title="Driver standings" span="lg:col-span-7">
        <motion.div variants={container} initial="hidden" animate="show" className="flex flex-col">
          <div className="grid grid-cols-[28px_1fr_auto] items-center gap-3 border-b border-line pb-2 text-[10px] font-semibold uppercase tracking-wide text-ink-faint sm:grid-cols-[28px_1fr_repeat(4,46px)]">
            <span>#</span>
            <span>Driver</span>
            <span className="hidden text-right sm:block">Wins</span>
            <span className="hidden text-right sm:block">Pod</span>
            <span className="hidden text-right sm:block">Pole</span>
            <span className="text-right">Pts</span>
          </div>
          {season.drivers.map((d, i) => (
            <motion.div
              key={d.code}
              variants={row}
              className="grid grid-cols-[28px_1fr_auto] items-center gap-3 border-b border-line/60 py-2 last:border-0 sm:grid-cols-[28px_1fr_repeat(4,46px)]"
            >
              <span className="text-[12px] font-bold text-ink-faint">{i + 1}</span>
              <div className="flex min-w-0 items-center gap-2">
                <span className="h-6 w-1 shrink-0 rounded-full" style={{ background: d.color }} />
                <div className="min-w-0">
                  <div className="truncate text-[13px] font-semibold text-ink">{d.name}</div>
                  <div className="truncate text-[10px] text-ink-faint">{d.team}</div>
                </div>
              </div>
              <span className="hidden text-right text-[12px] text-ink-dim sm:block">{d.wins}</span>
              <span className="hidden text-right text-[12px] text-ink-dim sm:block">{d.podiums}</span>
              <span className="hidden text-right text-[12px] text-ink-dim sm:block">{d.poles}</span>
              <span className="text-right text-[13px] font-bold text-ink">{Math.round(d.points)}</span>
            </motion.div>
          ))}
        </motion.div>
      </Card>

      <Card title="Constructor standings" span="lg:col-span-5">
        <motion.div variants={container} initial="hidden" animate="show" className="flex flex-col gap-3">
          {season.constructors.map((c, i) => (
            <motion.div key={c.team} variants={row}>
              <div className="mb-1 flex items-center gap-2">
                <span className="text-[12px] font-bold text-ink-faint">{i + 1}</span>
                <span className="truncate text-[13px] font-semibold text-ink">{c.team}</span>
                <span className="ml-auto text-[10px] text-ink-faint">{c.wins} wins</span>
                <span className="w-12 text-right text-[13px] font-bold text-ink">{Math.round(c.points)}</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-ov/[0.06]">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: c.color }}
                  initial={{ width: 0 }}
                  animate={{ width: `${(c.points / maxC) * 100}%` }}
                  transition={{ duration: 0.9, delay: 0.1 + i * 0.05, ease: [0.16, 1, 0.3, 1] }}
                />
              </div>
            </motion.div>
          ))}
          <p className="mt-1 text-[11px] text-ink-faint">
            Bars relative to leader ({Math.round(maxC)} pts) ·{" "}
            <span className="text-ink-dim">{season.year} season</span>. Driver leader holds {Math.round(maxD)} pts.
          </p>
        </motion.div>
      </Card>
    </div>
  );
}

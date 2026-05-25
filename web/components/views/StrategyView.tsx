"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/Card";
import { HeatmapCard } from "@/components/cards/HeatmapCard";
import { AnimatedNumber } from "@/components/ui/AnimatedNumber";
import type { Season } from "@/lib/types";

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.03, delayChildren: 0.05 } },
};
const rowV = { hidden: { opacity: 0, x: -10 }, show: { opacity: 1, x: 0 } };

export function StrategyView({ season }: { season: Season }) {
  const totalDnf = season.drivers.reduce((a, d) => a + d.dnfs, 0);
  const totalStarts = season.drivers.reduce((a, d) => a + d.racesEntered, 0);
  const finishRate = totalStarts ? Math.round(((totalStarts - totalDnf) / totalStarts) * 100) : 0;
  const totalFl = season.drivers.reduce((a, d) => a + d.fastestLaps, 0);

  const rel = [...season.drivers]
    .map((d) => ({
      ...d,
      finishPct: d.racesEntered ? Math.round(((d.racesEntered - d.dnfs) / d.racesEntered) * 100) : 0,
      ppr: d.racesEntered ? d.points / d.racesEntered : 0,
    }))
    .sort((a, b) => b.finishPct - a.finishPct || b.ppr - a.ppr);

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="grid grid-cols-1 gap-4 lg:grid-cols-12 lg:auto-rows-min"
    >
      <Card title="Season reliability" span="lg:col-span-12">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          {[
            { l: "Field finish rate", v: finishRate, suffix: "%", accent: "text-pos" },
            { l: "Total DNFs", v: totalDnf, suffix: "", accent: "text-f1" },
            { l: "Rounds run", v: season.totals.completedRounds, suffix: "", accent: "text-ink" },
            { l: "Fastest laps set", v: totalFl, suffix: "", accent: "text-ink" },
          ].map((s) => (
            <div key={s.l} className="rounded-xl bg-ov/[0.03] p-4">
              <div className="text-[11px] uppercase tracking-wide text-ink-faint">{s.l}</div>
              <AnimatedNumber value={s.v} suffix={s.suffix} className={`text-[26px] font-extrabold ${s.accent}`} />
            </div>
          ))}
        </div>
      </Card>

      <div className="lg:col-span-7">
        <HeatmapCard season={season} />
      </div>

      <Card title="Driver reliability" span="lg:col-span-5">
        <motion.div variants={container} className="flex flex-col">
          <div className="grid grid-cols-[1fr_repeat(3,52px)] gap-2 border-b border-line pb-2 text-[10px] font-semibold uppercase tracking-wide text-ink-faint">
            <span>Driver</span>
            <span className="text-right">Fin%</span>
            <span className="text-right">DNF</span>
            <span className="text-right">Pts/R</span>
          </div>
          {rel.map((d) => (
            <motion.div
              key={d.code}
              variants={rowV}
              className="grid grid-cols-[1fr_repeat(3,52px)] items-center gap-2 border-b border-line/60 py-2 last:border-0"
            >
              <div className="flex min-w-0 items-center gap-2">
                <span className="h-5 w-1 shrink-0 rounded-full" style={{ background: d.color }} />
                <span className="truncate text-[12px] font-semibold text-ink">{d.code}</span>
                <span className="truncate text-[10px] text-ink-faint">{d.team}</span>
              </div>
              <span className="text-right text-[12px] font-semibold text-pos">{d.finishPct}%</span>
              <span className="text-right text-[12px] text-ink-dim">{d.dnfs}</span>
              <span className="text-right text-[12px] font-bold text-ink">{d.ppr.toFixed(1)}</span>
            </motion.div>
          ))}
        </motion.div>
      </Card>
    </motion.div>
  );
}

"use client";

import { motion } from "framer-motion";
import { AreaChart } from "@/components/ui/AreaChart";
import type { Season, Driver } from "@/lib/types";

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.04, delayChildren: 0.05 } },
};
const item = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0 },
};

function DriverTile({ d, rank }: { d: Driver; rank: number }) {
  const numStr = d.number != null ? String(d.number) : "—";
  return (
    <motion.div
      variants={item}
      whileHover={{ y: -4 }}
      className="card overflow-hidden p-0"
    >
      <div
        className="relative flex items-center justify-between px-4 py-3"
        style={{ background: `linear-gradient(120deg, ${d.color}, transparent 140%)` }}
      >
        <div className="min-w-0">
          <div className="text-[11px] font-medium text-white/80">P{rank} · {d.team}</div>
          <div className="truncate text-[16px] font-bold text-white">{d.name}</div>
        </div>
        <div className="grid h-9 w-12 shrink-0 place-items-center rounded-lg bg-black/25 font-mono text-[14px] font-bold text-white">
          {numStr}
        </div>
      </div>

      <div className="px-4 pb-4 pt-3">
        <div className="flex items-end justify-between">
          <div>
            <div className="text-[10px] uppercase tracking-wide text-ink-faint">Points</div>
            <div className="text-[26px] font-extrabold leading-none text-ink">{Math.round(d.points)}</div>
          </div>
          <div className="grid grid-cols-3 gap-3 text-center">
            {[
              { l: "Wins", v: d.wins },
              { l: "Pod", v: d.podiums },
              { l: "Pole", v: d.poles },
            ].map((s) => (
              <div key={s.l}>
                <div className="text-[14px] font-bold text-ink">{s.v}</div>
                <div className="text-[9px] uppercase tracking-wide text-ink-faint">{s.l}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-2 h-[44px]">
          <AreaChart data={d.cumulative} color={d.color} height={44} showGrid={false} />
        </div>

        <div className="mt-1 flex items-center justify-between text-[10px] text-ink-faint">
          <span>{d.racesEntered} races</span>
          <span>{d.fastestLaps} FL · {d.dnfs} DNF</span>
        </div>
      </div>
    </motion.div>
  );
}

export function DriversView({ season }: { season: Season }) {
  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
    >
      {season.drivers.map((d, i) => (
        <DriverTile key={d.code} d={d} rank={i + 1} />
      ))}
    </motion.div>
  );
}

"use client";

import { motion } from "framer-motion";
import { driversByTeam } from "@/lib/season";
import type { Season } from "@/lib/types";

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.05, delayChildren: 0.05 } },
};
const item = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0 },
};

export function TeamsView({ season }: { season: Season }) {
  const teams = driversByTeam(season);
  const totalPoints = teams.reduce((a, t) => a + t.points, 0) || 1;
  const maxPoints = Math.max(...teams.map((t) => t.points), 1);

  return (
    <motion.div
      variants={container}
      initial="hidden"
      animate="show"
      className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3"
    >
      {teams.map((t, i) => {
        const share = ((t.points / totalPoints) * 100).toFixed(1);
        return (
          <motion.div key={t.team} variants={item} whileHover={{ y: -4 }} className="card overflow-hidden p-0">
            <div className="h-1.5 w-full" style={{ background: t.color }} />
            <div className="p-5">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-[12px] font-bold text-ink-faint">P{i + 1}</span>
                  <h3 className="text-[16px] font-bold text-ink">{t.team}</h3>
                </div>
                <span className="pill bg-ov/[0.06] text-ink-dim">{t.wins} wins</span>
              </div>

              <div className="mt-3 flex items-end gap-2">
                <span className="text-[30px] font-extrabold leading-none text-ink">{Math.round(t.points)}</span>
                <span className="mb-1 text-[12px] font-semibold text-ink-faint">pts · {share}% share</span>
              </div>

              <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-ov/[0.06]">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: t.color }}
                  initial={{ width: 0 }}
                  animate={{ width: `${(t.points / maxPoints) * 100}%` }}
                  transition={{ duration: 0.9, delay: 0.1 + i * 0.05, ease: [0.16, 1, 0.3, 1] }}
                />
              </div>

              <div className="mt-4 flex flex-col gap-2">
                {t.drivers.map((d) => (
                  <div key={d.code} className="flex items-center gap-2 rounded-xl bg-ov/[0.03] px-3 py-2">
                    <span className="grid h-7 w-9 place-items-center rounded-md text-[11px] font-bold text-white" style={{ background: d.color }}>
                      {d.code}
                    </span>
                    <span className="truncate text-[13px] font-medium text-ink">{d.name}</span>
                    <span className="ml-auto text-[13px] font-bold text-ink">{Math.round(d.points)}</span>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        );
      })}
    </motion.div>
  );
}

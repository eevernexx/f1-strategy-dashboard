"use client";

import { motion } from "framer-motion";
import { useState } from "react";
import { Card } from "@/components/ui/Card";
import type { Season } from "@/lib/types";

export function StandingsBarCard({ season }: { season: Season }) {
  const [mode, setMode] = useState<"drivers" | "teams">("drivers");

  const items =
    mode === "drivers"
      ? season.drivers.slice(0, 8).map((d) => ({ label: d.code, value: d.points, color: d.color }))
      : season.constructors.map((c) => ({
          label: c.team.split(" ")[0].slice(0, 3).toUpperCase(),
          value: c.points,
          color: c.color,
        }));

  const max = Math.max(...items.map((i) => i.value), 1);

  return (
    <Card
      title="Points standings"
      span="lg:col-span-8"
      action={
        <div className="flex items-center rounded-pill border border-line p-0.5 text-[11px] font-semibold">
          {(["drivers", "teams"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`relative rounded-pill px-2.5 py-1 capitalize transition-colors ${
                mode === m ? "text-white" : "text-ink-dim"
              }`}
            >
              {mode === m && (
                <motion.span
                  layoutId="bar-mode"
                  className="absolute inset-0 rounded-pill bg-white/[0.08]"
                />
              )}
              <span className="relative z-10">{m}</span>
            </button>
          ))}
        </div>
      }
    >
      <div className="flex h-[200px] items-end justify-between gap-2 px-1">
        {items.map((it, i) => (
          <div key={it.label + i} className="flex flex-1 flex-col items-center gap-2">
            <div className="relative flex h-[160px] w-full max-w-[34px] items-end">
              <motion.div
                className="w-full rounded-md"
                style={{
                  background: `linear-gradient(180deg, ${it.color}, ${it.color}55)`,
                }}
                initial={{ height: 0 }}
                animate={{ height: `${(it.value / max) * 100}%` }}
                transition={{ duration: 0.9, delay: i * 0.06, ease: [0.16, 1, 0.3, 1] }}
              >
                <span className="absolute -top-5 left-1/2 -translate-x-1/2 text-[10px] font-semibold text-ink-dim">
                  {Math.round(it.value)}
                </span>
              </motion.div>
            </div>
            <span className="text-[10px] font-medium text-ink-faint">{it.label}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}

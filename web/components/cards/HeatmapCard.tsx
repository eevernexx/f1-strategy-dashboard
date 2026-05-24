"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/Card";
import type { Season } from "@/lib/types";

function intensity(pts: number) {
  if (pts >= 25) return 1;
  if (pts >= 18) return 0.78;
  if (pts >= 10) return 0.55;
  if (pts > 0) return 0.32;
  return 0;
}

const LEGEND = [
  { label: "0", o: 0.08 },
  { label: ">1", o: 0.32 },
  { label: ">10", o: 0.55 },
  { label: ">18", o: 0.78 },
  { label: "win", o: 1 },
];

export function HeatmapCard({ season }: { season: Season }) {
  const drivers = season.drivers.slice(0, 8);
  const rounds = season.rounds;

  return (
    <Card
      title="Results heatmap"
      span="lg:col-span-8"
      action={
        <div className="hidden items-center gap-2 text-[10px] text-ink-faint sm:flex">
          {LEGEND.map((l) => (
            <span key={l.label} className="flex items-center gap-1">
              <span
                className="h-2.5 w-2.5 rounded-[3px]"
                style={{ background: `rgba(232,0,45,${l.o})` }}
              />
              {l.label}
            </span>
          ))}
        </div>
      }
    >
      <div className="flex flex-col gap-1.5">
        {drivers.map((d, r) => (
          <div key={d.code} className="flex items-center gap-2">
            <span className="w-8 shrink-0 text-[10px] font-semibold text-ink-dim">
              {d.code}
            </span>
            <div className="flex flex-1 gap-1">
              {rounds.map((rd, c) => {
                const pts = d.perRound[c] ?? 0;
                const o = intensity(pts);
                return (
                  <motion.div
                    key={rd.round}
                    title={`${rd.name}: ${pts} pts`}
                    className="h-5 flex-1 rounded-[4px]"
                    style={{
                      background:
                        o === 0
                          ? "rgba(255,255,255,0.05)"
                          : `${d.color}${Math.round(o * 255)
                              .toString(16)
                              .padStart(2, "0")}`,
                    }}
                    initial={{ opacity: 0, scale: 0.6 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.2 + (r * rounds.length + c) * 0.004 }}
                  />
                );
              })}
            </div>
          </div>
        ))}
        <div className="mt-1 flex items-center gap-2">
          <span className="w-8 shrink-0" />
          <div className="flex flex-1 justify-between text-[9px] text-ink-faint">
            <span>R1</span>
            <span>R{Math.round(rounds.length / 2)}</span>
            <span>R{rounds.length}</span>
          </div>
        </div>
      </div>
    </Card>
  );
}

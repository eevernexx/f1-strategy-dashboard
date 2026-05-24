"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/Card";
import type { Driver } from "@/lib/types";

function hexToRgba(hex: string, a: number) {
  const m = hex.replace("#", "");
  const r = parseInt(m.slice(0, 2), 16);
  const g = parseInt(m.slice(2, 4), 16);
  const b = parseInt(m.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}

export function DriverCard({ driver }: { driver: Driver }) {
  const num = driver.number ?? 0;
  const numStr = String(num).padStart(2, "0");

  return (
    <Card title="Featured driver" showArrow span="lg:col-span-3">
      <motion.div
        whileHover={{ rotateX: 6, rotateY: -6, scale: 1.02 }}
        transition={{ type: "spring", stiffness: 200, damping: 18 }}
        style={{
          transformStyle: "preserve-3d",
          background: `linear-gradient(135deg, ${hexToRgba(driver.color, 0.95)}, ${hexToRgba(
            driver.color,
            0.55,
          )} 55%, rgba(20,20,28,0.95))`,
        }}
        className="relative overflow-hidden rounded-2xl p-4 shadow-lg"
      >
        <div className="absolute -right-6 -top-8 h-28 w-28 rounded-full bg-white/15 blur-md" />
        <div className="flex items-start justify-between">
          <div>
            <div className="text-[12px] font-medium text-white/80">{driver.team}</div>
            <div className="text-[17px] font-bold text-white">{driver.name}</div>
          </div>
          <div className="grid h-7 w-9 place-items-center rounded-md bg-white/25 text-[11px] font-bold text-white">
            #{numStr}
          </div>
        </div>

        <div className="mt-6 font-mono text-[17px] tracking-[0.18em] text-white drop-shadow">
          {driver.code} · {String(driver.points).padStart(4, "0")} PTS
        </div>

        <div className="mt-4 flex items-end justify-between">
          <div>
            <div className="text-[9px] uppercase tracking-wider text-white/60">Wins / Poles</div>
            <div className="text-[13px] font-semibold text-white">
              {driver.wins} / {driver.poles}
            </div>
          </div>
          <div className="text-right">
            <div className="text-[9px] uppercase tracking-wider text-white/60">Podiums</div>
            <div className="text-[13px] font-semibold text-white">{driver.podiums}</div>
          </div>
        </div>
      </motion.div>

      <div className="mt-3 grid grid-cols-2 gap-2">
        <button className="chip-btn justify-center">Profile</button>
        <button className="chip-btn justify-center">Compare</button>
      </div>
    </Card>
  );
}

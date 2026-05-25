"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Card } from "@/components/ui/Card";
import { AreaChart } from "@/components/ui/AreaChart";
import type { Driver } from "@/lib/types";
import type { ViewId } from "@/lib/nav";

function hexToRgba(hex: string, a: number) {
  const m = hex.replace("#", "");
  const r = parseInt(m.slice(0, 2), 16);
  const g = parseInt(m.slice(2, 4), 16);
  const b = parseInt(m.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${a})`;
}

export function DriverCard({
  driver,
  onNavigate,
}: {
  driver: Driver;
  onNavigate?: (v: ViewId) => void;
}) {
  const [open, setOpen] = useState(false);
  const num = driver.number ?? 0;
  const numStr = String(num).padStart(2, "0");

  return (
    <Card title="Featured driver" span="lg:col-span-4">
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
        <button onClick={() => setOpen(true)} className="chip-btn justify-center">
          Profile
        </button>
        <button onClick={() => onNavigate?.("telemetry")} className="chip-btn justify-center">
          Compare
        </button>
      </div>

      <AnimatePresence>
        {open && <ProfileModal driver={driver} onClose={() => setOpen(false)} />}
      </AnimatePresence>
    </Card>
  );
}

function ProfileModal({ driver, onClose }: { driver: Driver; onClose: () => void }) {
  const stats = [
    { l: "Points", v: Math.round(driver.points) },
    { l: "Wins", v: driver.wins },
    { l: "Podiums", v: driver.podiums },
    { l: "Poles", v: driver.poles },
    { l: "Fastest laps", v: driver.fastestLaps },
    { l: "DNFs", v: driver.dnfs },
    { l: "Races", v: driver.racesEntered },
    {
      l: "Pts / race",
      v: driver.racesEntered ? (driver.points / driver.racesEntered).toFixed(1) : "0",
    },
  ];

  return (
    <div className="fixed inset-0 z-50 grid place-items-center p-4">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
      />
      <motion.div
        initial={{ opacity: 0, y: 24, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 24, scale: 0.96 }}
        transition={{ type: "spring", stiffness: 280, damping: 26 }}
        className="card relative w-full max-w-md p-0"
      >
        <div
          className="flex items-center justify-between rounded-t-card px-5 py-4"
          style={{ background: `linear-gradient(120deg, ${driver.color}, transparent 150%)` }}
        >
          <div>
            <div className="text-[12px] font-medium text-white/80">{driver.team}</div>
            <div className="text-[20px] font-bold text-white">{driver.name}</div>
          </div>
          <div className="grid h-9 w-12 place-items-center rounded-lg bg-black/25 font-mono text-[14px] font-bold text-white">
            {driver.number ?? "—"}
          </div>
        </div>

        <div className="p-5">
          <div className="grid grid-cols-4 gap-3">
            {stats.map((s) => (
              <div key={s.l} className="rounded-xl bg-ov/[0.03] p-2.5 text-center">
                <div className="text-[16px] font-extrabold text-ink">{s.v}</div>
                <div className="text-[9px] uppercase tracking-wide text-ink-faint">{s.l}</div>
              </div>
            ))}
          </div>

          <div className="mt-4 text-[10px] uppercase tracking-wide text-ink-faint">
            Cumulative points
          </div>
          <div className="mt-1 h-[80px]">
            <AreaChart data={driver.cumulative} color={driver.color} height={80} showGrid={false} />
          </div>

          <button
            onClick={onClose}
            className="mt-4 w-full rounded-pill bg-f1 py-2.5 text-[13px] font-semibold text-white transition-colors hover:bg-f1-hover"
          >
            Close
          </button>
        </div>
      </motion.div>
    </div>
  );
}

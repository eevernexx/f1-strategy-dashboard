"use client";

import { motion } from "framer-motion";

interface Seg {
  label: string;
  value: number;
  color: string;
}

// Nested concentric arcs (Finrise "Top spending" style). Each segment is its
// own ring, length proportional to its share of the max segment.
export function RingChart({
  segments,
  size = 188,
}: {
  segments: Seg[];
  size?: number;
}) {
  const cx = size / 2;
  const cy = size / 2;
  const stroke = 9;
  const gap = 5;
  const top = segments.slice(0, 5);
  const max = Math.max(...top.map((s) => s.value), 1);
  const total = segments.reduce((a, s) => a + s.value, 0);
  const startR = size / 2 - 8;

  return (
    <div className="relative grid place-items-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        {top.map((s, i) => {
          const r = startR - i * (stroke + gap);
          const c = 2 * Math.PI * r;
          const frac = (s.value / max) * 0.82; // leave a tail gap, like the ref
          return (
            <g key={s.label}>
              <circle
                cx={cx}
                cy={cy}
                r={r}
                fill="none"
                stroke="rgba(255,255,255,0.05)"
                strokeWidth={stroke}
              />
              <motion.circle
                cx={cx}
                cy={cy}
                r={r}
                fill="none"
                stroke={s.color}
                strokeWidth={stroke}
                strokeLinecap="round"
                strokeDasharray={c}
                initial={{ strokeDashoffset: c }}
                animate={{ strokeDashoffset: c - frac * c }}
                transition={{ duration: 1.2, delay: 0.2 + i * 0.12, ease: [0.16, 1, 0.3, 1] }}
              />
            </g>
          );
        })}
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="text-[11px] font-medium text-ink-faint">Total</span>
        <span className="text-2xl font-extrabold tracking-tight text-ink">
          {Math.round(total)}
        </span>
        <span className="text-[11px] text-ink-dim">points</span>
      </div>
    </div>
  );
}

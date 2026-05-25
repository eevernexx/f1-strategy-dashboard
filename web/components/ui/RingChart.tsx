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
  size = 200,
}: {
  segments: Seg[];
  size?: number;
}) {
  const cx = size / 2;
  const cy = size / 2;
  const stroke = 8;
  const gap = 5;
  const top = segments.slice(0, 5);
  const max = Math.max(...top.map((s) => s.value), 1);
  const total = segments.reduce((a, s) => a + s.value, 0);
  const startR = size / 2 - 8;

  // innermost ring radius → clear hole so the centre label never overlaps arcs
  const innerR = startR - (top.length - 1) * (stroke + gap);
  const holeR = Math.max(innerR - stroke / 2 - 2, 24);

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
                stroke="rgb(var(--c-ov) / 0.06)"
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
        {/* backing disc masks any arc behind the centre label */}
        <circle cx={cx} cy={cy} r={holeR} fill="var(--c-card)" />
      </svg>
      <div
        className="absolute flex flex-col items-center text-center"
        style={{ maxWidth: holeR * 2 - 6 }}
      >
        <span className="text-[10px] font-medium uppercase tracking-wide text-ink-faint">
          Total
        </span>
        <span className="text-[22px] font-extrabold leading-none tracking-tight text-ink">
          {Math.round(total)}
        </span>
        <span className="text-[10px] text-ink-dim">points</span>
      </div>
    </div>
  );
}

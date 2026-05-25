"use client";

import { motion } from "framer-motion";
import { useId } from "react";
import { smoothPath } from "@/lib/format";

export interface LineSeries {
  label: string;
  color: string;
  data: number[];
}

export function MultiLineChart({
  series,
  height = 260,
  width = 760,
  labels,
}: {
  series: LineSeries[];
  height?: number;
  width?: number;
  labels?: string[];
}) {
  const id = useId().replace(/:/g, "");
  const pad = { top: 16, right: 14, bottom: 26, left: 36 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const allVals = series.flatMap((s) => s.data);
  const max = Math.max(...allVals, 1);
  const min = Math.min(...allVals, 0);
  const range = max - min || 1;
  const len = Math.max(...series.map((s) => s.data.length), 1);

  const x = (i: number) => pad.left + (i / (len - 1 || 1)) * innerW;
  const y = (v: number) => pad.top + innerH - ((v - min) / range) * innerH;

  const gridY = [0, 0.25, 0.5, 0.75, 1];

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-full w-full">
      {gridY.map((f, i) => {
        const gy = pad.top + innerH * f;
        return (
          <g key={i}>
            <line x1={pad.left} x2={width - pad.right} y1={gy} y2={gy} stroke="rgb(var(--c-ov) / 0.06)" strokeWidth="1" />
            <text x={pad.left - 8} y={gy + 3} fill="var(--c-ink-faint)" fontSize="9" textAnchor="end">
              {Math.round(max - (max - min) * f)}
            </text>
          </g>
        );
      })}

      {series.map((s, si) => {
        const pts: [number, number][] = s.data.map((v, i) => [x(i), y(v)]);
        const line = smoothPath(pts);
        const last = pts[pts.length - 1];
        return (
          <g key={s.label}>
            <motion.path
              d={line}
              fill="none"
              stroke={s.color}
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 1.2, delay: 0.15 + si * 0.08, ease: [0.16, 1, 0.3, 1] }}
            />
            <motion.circle
              cx={last[0]}
              cy={last[1]}
              r="3.2"
              fill={s.color}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 1.1 + si * 0.05, type: "spring", stiffness: 400 }}
            />
          </g>
        );
      })}

      {labels &&
        labels.map((lb, i) => (
          <text
            key={i}
            x={x(Math.round((i / (labels.length - 1 || 1)) * (len - 1)))}
            y={height - 8}
            fill="var(--c-ink-faint)"
            fontSize="9"
            textAnchor="middle"
          >
            {lb}
          </text>
        ))}
      <defs>
        <linearGradient id={`ml-${id}`} />
      </defs>
    </svg>
  );
}

"use client";

import { motion } from "framer-motion";
import { useId } from "react";
import { smoothPath } from "@/lib/format";

export function AreaChart({
  data,
  color = "#E8002D",
  height = 150,
  width = 520,
  showGrid = true,
  labels,
}: {
  data: number[];
  color?: string;
  height?: number;
  width?: number;
  showGrid?: boolean;
  labels?: string[];
}) {
  const id = useId().replace(/:/g, "");
  const pad = { top: 14, right: 8, bottom: 22, left: 8 };
  const w = width;
  const h = height;
  const innerW = w - pad.left - pad.right;
  const innerH = h - pad.top - pad.bottom;

  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;

  const pts: [number, number][] = data.map((v, i) => [
    pad.left + (i / (data.length - 1 || 1)) * innerW,
    pad.top + innerH - ((v - min) / range) * innerH,
  ]);

  const line = smoothPath(pts);
  const area = `${line} L ${pts[pts.length - 1][0]},${pad.top + innerH} L ${pts[0][0]},${pad.top + innerH} Z`;

  const gridY = [0.25, 0.5, 0.75, 1].map((f) => pad.top + innerH * f);

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-full w-full" preserveAspectRatio="none">
      <defs>
        <linearGradient id={`fill-${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.32" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>

      {showGrid &&
        gridY.map((y, i) => (
          <line
            key={i}
            x1={pad.left}
            x2={w - pad.right}
            y1={y}
            y2={y}
            stroke="rgba(255,255,255,0.05)"
            strokeWidth="1"
          />
        ))}

      <motion.path
        d={area}
        fill={`url(#fill-${id})`}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8, delay: 0.5 }}
      />
      <motion.path
        d={line}
        fill="none"
        stroke={color}
        strokeWidth="2.4"
        strokeLinecap="round"
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 1.3, ease: [0.16, 1, 0.3, 1] }}
      />
      {/* end dot */}
      <motion.circle
        cx={pts[pts.length - 1][0]}
        cy={pts[pts.length - 1][1]}
        r="3.5"
        fill={color}
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ delay: 1.4, type: "spring", stiffness: 400 }}
      />

      {labels &&
        labels.map((lb, i) => (
          <text
            key={i}
            x={pad.left + (i / (labels.length - 1 || 1)) * innerW}
            y={h - 6}
            fill="rgba(255,255,255,0.35)"
            fontSize="10"
            textAnchor="middle"
          >
            {lb}
          </text>
        ))}
    </svg>
  );
}

"use client";

import { Card } from "@/components/ui/Card";
import { RingChart } from "@/components/ui/RingChart";
import type { Season } from "@/lib/types";

export function TeamPointsCard({ season }: { season: Season }) {
  const top = season.constructors.slice(0, 5);
  const segments = top.map((c) => ({
    label: c.team,
    value: c.points,
    color: c.color,
  }));

  return (
    <Card title="Constructor points" span="lg:col-span-3">
      <div className="flex flex-col items-center">
        <RingChart segments={segments} />
        <div className="mt-3 grid w-full grid-cols-2 gap-x-3 gap-y-1.5">
          {top.map((c) => (
            <div key={c.team} className="flex items-center gap-1.5 text-[11px]">
              <span
                className="h-2 w-2 shrink-0 rounded-full"
                style={{ background: c.color }}
              />
              <span className="truncate text-ink-dim">{c.team}</span>
              <span className="ml-auto font-semibold text-ink">{Math.round(c.points)}</span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}

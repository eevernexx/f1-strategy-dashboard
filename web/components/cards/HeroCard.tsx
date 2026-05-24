"use client";

import { Card } from "@/components/ui/Card";
import { AreaChart } from "@/components/ui/AreaChart";
import { AnimatedNumber } from "@/components/ui/AnimatedNumber";
import type { Season } from "@/lib/types";

export function HeroCard({ season }: { season: Season }) {
  const leader = season.drivers[0];
  const second = season.drivers[1];
  const gap = leader.points - (second?.points ?? 0);
  const winRate = leader.racesEntered
    ? Math.round((leader.wins / leader.racesEntered) * 100)
    : 0;
  const labels = season.rounds
    .filter((_, i) => i % Math.ceil(season.rounds.length / 6) === 0)
    .map((r) => r.circuit.split(" ")[0].slice(0, 3) || `R${r.round}`);

  return (
    <Card showArrow span="lg:col-span-6" bodyClassName="flex flex-col">
      <div className="mb-1 flex items-center gap-2 text-[13px] font-medium text-ink-dim">
        <span
          className="h-2.5 w-2.5 rounded-full"
          style={{ background: leader.color }}
        />
        Championship leader · {leader.name}
      </div>

      <div className="flex items-end gap-3">
        <AnimatedNumber
          value={leader.points}
          className="text-5xl font-extrabold tracking-tight text-ink md:text-[56px]"
        />
        <span className="mb-2 text-lg font-semibold text-ink-faint">PTS</span>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <span className="pill bg-f1-soft text-f1">
          {leader.wins} wins
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
            <path d="M7 17 17 7M17 7H9M17 7v8" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
        <span className="pill bg-white/[0.05] text-ink-dim">+{gap.toFixed(0)} pt lead</span>
        <span className="pill bg-white/[0.05] text-ink-dim">{winRate}% win rate</span>
      </div>

      <div className="relative mt-4 flex-1">
        <div className="absolute right-0 top-0 flex flex-col gap-1 text-[10px] text-ink-faint">
          <span>{Math.round(leader.points)}</span>
          <span>{Math.round(leader.points * 0.66)}</span>
          <span>{Math.round(leader.points * 0.33)}</span>
        </div>
        <div className="h-[150px]">
          <AreaChart
            data={leader.cumulative}
            color={leader.color}
            labels={labels}
          />
        </div>
      </div>
    </Card>
  );
}

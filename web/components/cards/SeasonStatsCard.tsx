"use client";

import { Card } from "@/components/ui/Card";
import { AreaChart } from "@/components/ui/AreaChart";
import { AnimatedNumber } from "@/components/ui/AnimatedNumber";
import type { Season } from "@/lib/types";

export function SeasonStatsCard({ season }: { season: Season }) {
  const totalDnf = season.drivers.reduce((a, d) => a + d.dnfs, 0);
  const totalStarts = season.drivers.reduce((a, d) => a + d.racesEntered, 0);
  const finishRate = totalStarts
    ? Math.round(((totalStarts - totalDnf) / totalStarts) * 100)
    : 0;
  const leader = season.drivers[0];

  return (
    <Card title="Reliability & momentum" span="lg:col-span-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="text-[11px] font-medium text-ink-faint">Rounds</div>
          <div className="flex items-baseline gap-1">
            <AnimatedNumber
              value={season.totals.completedRounds}
              className="text-2xl font-extrabold text-ink"
            />
            <span className="text-[12px] font-semibold text-pos">
              of {season.totals.rounds}
            </span>
          </div>
        </div>
        <div>
          <div className="text-[11px] font-medium text-ink-faint">Finish rate</div>
          <AnimatedNumber
            value={finishRate}
            suffix="%"
            className="text-2xl font-extrabold text-ink"
          />
        </div>
        <div>
          <div className="text-[11px] font-medium text-ink-faint">Total DNFs</div>
          <AnimatedNumber value={totalDnf} className="text-2xl font-extrabold text-f1" />
        </div>
      </div>

      <div className="mt-3 h-[110px]">
        <AreaChart data={leader.perRound} color="#f4f4f6" showGrid={false} height={110} />
      </div>
    </Card>
  );
}

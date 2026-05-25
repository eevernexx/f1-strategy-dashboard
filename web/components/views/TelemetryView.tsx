"use client";

import { useState } from "react";
import { Card } from "@/components/ui/Card";
import { MultiLineChart, type LineSeries } from "@/components/ui/MultiLineChart";
import type { Season } from "@/lib/types";

export function TelemetryView({ season }: { season: Season }) {
  const [selected, setSelected] = useState<string[]>(
    season.drivers.slice(0, 5).map((d) => d.code),
  );

  const toggle = (code: string) =>
    setSelected((prev) =>
      prev.includes(code) ? prev.filter((c) => c !== code) : [...prev, code],
    );

  const chosen = season.drivers.filter((d) => selected.includes(d.code));
  const cumSeries: LineSeries[] = chosen.map((d) => ({
    label: d.code,
    color: d.color,
    data: d.cumulative,
  }));
  const perRoundSeries: LineSeries[] = chosen.map((d) => ({
    label: d.code,
    color: d.color,
    data: d.perRound,
  }));

  const labels = season.rounds
    .filter((_, i) => i % Math.ceil(season.rounds.length / 8) === 0)
    .map((r) => `R${r.round}`);

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
      <Card title="Select drivers" span="lg:col-span-12">
        <div className="flex flex-wrap gap-2">
          {season.drivers.map((d) => {
            const on = selected.includes(d.code);
            return (
              <button
                key={d.code}
                onClick={() => toggle(d.code)}
                className="flex items-center gap-1.5 rounded-pill border px-3 py-1.5 text-[12px] font-semibold transition-colors"
                style={{
                  borderColor: on ? d.color : "var(--c-line)",
                  background: on ? `${d.color}22` : "transparent",
                  color: on ? "var(--c-ink)" : "var(--c-ink-dim)",
                }}
              >
                <span className="h-2 w-2 rounded-full" style={{ background: d.color }} />
                {d.code}
              </button>
            );
          })}
        </div>
      </Card>

      <Card title="Cumulative points race-by-race" span="lg:col-span-7">
        <div className="h-[300px]">
          {cumSeries.length > 0 ? (
            <MultiLineChart series={cumSeries} labels={labels} height={300} />
          ) : (
            <Empty />
          )}
        </div>
      </Card>

      <Card title="Per-round points (momentum)" span="lg:col-span-5">
        <div className="h-[300px]">
          {perRoundSeries.length > 0 ? (
            <MultiLineChart series={perRoundSeries} labels={labels} height={300} />
          ) : (
            <Empty />
          )}
        </div>
      </Card>

      <Card title="Comparison" span="lg:col-span-12">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {chosen.map((d) => {
            const avg = d.racesEntered ? d.points / d.racesEntered : 0;
            return (
              <div key={d.code} className="rounded-xl bg-ov/[0.03] p-3">
                <div className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ background: d.color }} />
                  <span className="text-[13px] font-bold text-ink">{d.code}</span>
                </div>
                <div className="mt-2 text-[20px] font-extrabold text-ink">{avg.toFixed(1)}</div>
                <div className="text-[10px] uppercase tracking-wide text-ink-faint">pts / race</div>
                <div className="mt-1 text-[11px] text-ink-dim">{d.wins} wins · {d.dnfs} DNF</div>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}

function Empty() {
  return (
    <div className="grid h-full place-items-center text-[13px] text-ink-faint">
      Select at least one driver above.
    </div>
  );
}

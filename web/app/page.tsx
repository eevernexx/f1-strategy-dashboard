"use client";

import { useEffect, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { MobileNav } from "@/components/MobileNav";
import { TopBar } from "@/components/TopBar";
import { OverviewView } from "@/components/views/OverviewView";
import { StandingsView } from "@/components/views/StandingsView";
import { DriversView } from "@/components/views/DriversView";
import { TeamsView } from "@/components/views/TeamsView";
import { CalendarView } from "@/components/views/CalendarView";
import { TelemetryView } from "@/components/views/TelemetryView";
import { StrategyView } from "@/components/views/StrategyView";
import { NAV, type ViewId } from "@/lib/nav";
import type { DataIndex, Season } from "@/lib/types";

export default function Page() {
  const [index, setIndex] = useState<DataIndex | null>(null);
  const [year, setYear] = useState<number | null>(null);
  const [season, setSeason] = useState<Season | null>(null);
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState<ViewId>("overview");
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    fetch("/data/index.json")
      .then((r) => r.json())
      .then((idx: DataIndex) => {
        setIndex(idx);
        setYear(idx.default);
      });
  }, []);

  useEffect(() => {
    if (year == null) return;
    setLoading(true);
    fetch(`/data/${year}.json`)
      .then((r) => r.json())
      .then((s: Season) => {
        setSeason(s);
        setLoading(false);
      });
  }, [year]);

  const viewLabel = NAV.find((n) => n.id === view)?.label ?? "Overview";

  return (
    <div className="flex min-h-screen">
      <Sidebar active={view} onChange={setView} />
      <MobileNav open={menuOpen} active={view} onChange={setView} onClose={() => setMenuOpen(false)} />

      <main className="flex-1 px-4 py-5 md:px-7 md:py-6">
        <div className="mx-auto max-w-[1320px]">
          <TopBar
            years={index?.years ?? []}
            year={year ?? 0}
            onYearChange={setYear}
            viewLabel={viewLabel}
            onMenu={() => setMenuOpen(true)}
          />

          {loading || !season ? (
            <SkeletonGrid />
          ) : (
            <ActiveView view={view} season={season} onNavigate={setView} />
          )}
        </div>
      </main>
    </div>
  );
}

function ActiveView({
  view,
  season,
  onNavigate,
}: {
  view: ViewId;
  season: Season;
  onNavigate: (v: ViewId) => void;
}) {
  switch (view) {
    case "overview":
      return <OverviewView season={season} onNavigate={onNavigate} />;
    case "standings":
      return <StandingsView season={season} />;
    case "drivers":
      return <DriversView season={season} />;
    case "teams":
      return <TeamsView season={season} />;
    case "calendar":
      return <CalendarView season={season} />;
    case "telemetry":
      return <TelemetryView season={season} />;
    case "strategy":
      return <StrategyView season={season} />;
    default:
      return <OverviewView season={season} />;
  }
}

function SkeletonGrid() {
  const spans = [
    "lg:col-span-5",
    "lg:col-span-3",
    "lg:col-span-4",
    "lg:col-span-8",
    "lg:col-span-4 lg:row-span-2",
    "lg:col-span-8",
    "lg:col-span-6",
    "lg:col-span-6",
  ];
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-12 lg:auto-rows-min">
      {spans.map((s, i) => (
        <div
          key={i}
          className={`card h-48 animate-pulse bg-ov/[0.03] ${s}`}
          style={{ animationDelay: `${i * 60}ms` }}
        />
      ))}
    </div>
  );
}

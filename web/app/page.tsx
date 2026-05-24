"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Sidebar } from "@/components/Sidebar";
import { TopBar } from "@/components/TopBar";
import { HeroCard } from "@/components/cards/HeroCard";
import { TeamPointsCard } from "@/components/cards/TeamPointsCard";
import { DriverCard } from "@/components/cards/DriverCard";
import { StandingsBarCard } from "@/components/cards/StandingsBarCard";
import { HeatmapCard } from "@/components/cards/HeatmapCard";
import { RecentResultsCard } from "@/components/cards/RecentResultsCard";
import { SeasonStatsCard } from "@/components/cards/SeasonStatsCard";
import { ContendersCard } from "@/components/cards/ContendersCard";
import type { DataIndex, Season } from "@/lib/types";

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.07, delayChildren: 0.1 } },
};

export default function Page() {
  const [index, setIndex] = useState<DataIndex | null>(null);
  const [year, setYear] = useState<number | null>(null);
  const [season, setSeason] = useState<Season | null>(null);
  const [loading, setLoading] = useState(true);

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

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 px-4 py-5 md:px-7 md:py-6">
        <div className="mx-auto max-w-[1320px]">
          <TopBar
            years={index?.years ?? []}
            year={year ?? 0}
            onYearChange={(y) => setYear(y)}
          />

          {loading || !season ? (
            <SkeletonGrid />
          ) : (
            <motion.div
              key={season.year}
              variants={container}
              initial="hidden"
              animate="show"
              className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-12 lg:auto-rows-min"
            >
              <HeroCard season={season} />
              <TeamPointsCard season={season} />
              <DriverCard driver={season.drivers[0]} />
              <StandingsBarCard season={season} />
              <RecentResultsCard season={season} />
              <HeatmapCard season={season} />
              <SeasonStatsCard season={season} />
              <ContendersCard season={season} />
            </motion.div>
          )}
        </div>
      </main>
    </div>
  );
}

function SkeletonGrid() {
  const spans = [
    "lg:col-span-6",
    "lg:col-span-3",
    "lg:col-span-3",
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
          className={`card h-48 animate-pulse bg-white/[0.03] ${s}`}
          style={{ animationDelay: `${i * 60}ms` }}
        />
      ))}
    </div>
  );
}

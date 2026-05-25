"use client";

import { motion } from "framer-motion";
import { HeroCard } from "@/components/cards/HeroCard";
import { TeamPointsCard } from "@/components/cards/TeamPointsCard";
import { DriverCard } from "@/components/cards/DriverCard";
import { StandingsBarCard } from "@/components/cards/StandingsBarCard";
import { HeatmapCard } from "@/components/cards/HeatmapCard";
import { RecentResultsCard } from "@/components/cards/RecentResultsCard";
import { SeasonStatsCard } from "@/components/cards/SeasonStatsCard";
import { ContendersCard } from "@/components/cards/ContendersCard";
import type { Season } from "@/lib/types";
import type { ViewId } from "@/lib/nav";

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06, delayChildren: 0.08 } },
};

export function OverviewView({
  season,
  onNavigate,
}: {
  season: Season;
  onNavigate?: (v: ViewId) => void;
}) {
  return (
    <motion.div
      key={season.year}
      variants={container}
      initial="hidden"
      animate="show"
      className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-12 lg:auto-rows-min"
    >
      <HeroCard season={season} />
      <TeamPointsCard season={season} />
      <DriverCard driver={season.drivers[0]} onNavigate={onNavigate} />
      <StandingsBarCard season={season} />
      <RecentResultsCard season={season} />
      <HeatmapCard season={season} />
      <SeasonStatsCard season={season} />
      <ContendersCard season={season} onNavigate={onNavigate} />
    </motion.div>
  );
}

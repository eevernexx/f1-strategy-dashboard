import type { Season, Driver } from "@/lib/types";

export function driversByTeam(season: Season): { team: string; color: string; points: number; wins: number; drivers: Driver[] }[] {
  const map = new Map<string, Driver[]>();
  for (const d of season.drivers) {
    const arr = map.get(d.team) ?? [];
    arr.push(d);
    map.set(d.team, arr);
  }
  return season.constructors.map((c) => ({
    team: c.team,
    color: c.color,
    points: c.points,
    wins: c.wins,
    drivers: (map.get(c.team) ?? []).sort((a, b) => b.points - a.points),
  }));
}

// Top scorer of a given round (≈ winner, 25 pts) — used for the calendar.
export function roundTopScorer(season: Season, roundIndex: number): Driver | null {
  let best: Driver | null = null;
  let bestPts = -1;
  for (const d of season.drivers) {
    const pts = d.perRound[roundIndex] ?? 0;
    if (pts > bestPts) {
      bestPts = pts;
      best = d;
    }
  }
  return bestPts > 0 ? best : null;
}

export interface RoundMeta {
  round: number;
  name: string;
  circuit: string;
}

export interface Driver {
  code: string;
  name: string;
  team: string;
  color: string;
  number: number | null;
  points: number;
  wins: number;
  podiums: number;
  poles: number;
  fastestLaps: number;
  dnfs: number;
  racesEntered: number;
  perRound: number[];
  cumulative: number[];
}

export interface Constructor {
  team: string;
  color: string;
  points: number;
  wins: number;
}

export interface RecentResult {
  round: number;
  gp: string;
  code: string;
  name: string;
  team: string;
  color: string;
  position: number | null;
  points: number;
  status: "Success" | "Failure";
  result: "Win" | "Podium" | "Points" | "Finished" | "DNF";
}

export interface SeasonTotals {
  drivers: number;
  rounds: number;
  totalPoints: number;
  completedRounds: number;
}

export interface Season {
  year: number;
  rounds: RoundMeta[];
  drivers: Driver[];
  constructors: Constructor[];
  recent: RecentResult[];
  totals: SeasonTotals;
}

export interface DataIndex {
  years: number[];
  default: number;
}

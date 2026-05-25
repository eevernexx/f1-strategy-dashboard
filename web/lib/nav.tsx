import type { ReactElement } from "react";

export type ViewId =
  | "overview"
  | "telemetry"
  | "standings"
  | "drivers"
  | "strategy"
  | "calendar"
  | "teams";

type IconProps = { className?: string };
type Icon = (p: IconProps) => ReactElement;

const Home: Icon = ({ className }) => (
  <svg className={className} width="20" height="20" viewBox="0 0 24 24" fill="none">
    <path d="M3 10.5 12 3l9 7.5M5 9.5V20a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V9.5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);
const Gauge: Icon = ({ className }) => (
  <svg className={className} width="20" height="20" viewBox="0 0 24 24" fill="none">
    <path d="M12 13l4-4M21 12a9 9 0 1 0-18 0" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    <circle cx="12" cy="13" r="1.6" fill="currentColor" />
  </svg>
);
const Chart: Icon = ({ className }) => (
  <svg className={className} width="20" height="20" viewBox="0 0 24 24" fill="none">
    <path d="M4 19V5M4 19h16M8 16l3-4 3 2 4-6" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);
const Cards: Icon = ({ className }) => (
  <svg className={className} width="20" height="20" viewBox="0 0 24 24" fill="none">
    <rect x="3" y="6" width="18" height="12" rx="2.5" stroke="currentColor" strokeWidth="1.7" />
    <path d="M3 10h18" stroke="currentColor" strokeWidth="1.7" />
  </svg>
);
const Flag: Icon = ({ className }) => (
  <svg className={className} width="20" height="20" viewBox="0 0 24 24" fill="none">
    <path d="M5 21V4m0 1h11l-1.5 3L16 11H5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);
const Calendar: Icon = ({ className }) => (
  <svg className={className} width="20" height="20" viewBox="0 0 24 24" fill="none">
    <rect x="3.5" y="5" width="17" height="16" rx="2.5" stroke="currentColor" strokeWidth="1.7" />
    <path d="M3.5 9.5h17M8 3v4M16 3v4" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
  </svg>
);
const People: Icon = ({ className }) => (
  <svg className={className} width="20" height="20" viewBox="0 0 24 24" fill="none">
    <circle cx="9" cy="8" r="3" stroke="currentColor" strokeWidth="1.7" />
    <path d="M3.5 20a5.5 5.5 0 0 1 11 0M16 6.5a3 3 0 0 1 0 5.6M21 20a5 5 0 0 0-3.5-4.8" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
  </svg>
);

export const NAV: { id: ViewId; label: string; icon: Icon }[] = [
  { id: "overview", label: "Overview", icon: Home },
  { id: "telemetry", label: "Pace & Momentum", icon: Gauge },
  { id: "standings", label: "Standings", icon: Chart },
  { id: "drivers", label: "Drivers", icon: Cards },
  { id: "strategy", label: "Reliability", icon: Flag },
  { id: "calendar", label: "Calendar", icon: Calendar },
  { id: "teams", label: "Teams", icon: People },
];

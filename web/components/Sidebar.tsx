"use client";

import { motion } from "framer-motion";
import { useState, type ReactElement } from "react";

type Icon = (p: { className?: string }) => ReactElement;

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
const Cog: Icon = ({ className }) => (
  <svg className={className} width="20" height="20" viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.7" />
    <path d="M12 2v2.5M12 19.5V22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M2 12h2.5M19.5 12H22M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
  </svg>
);
const Logout: Icon = ({ className }) => (
  <svg className={className} width="20" height="20" viewBox="0 0 24 24" fill="none">
    <path d="M14 7V5a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2v-2M9 12h12m0 0-3-3m3 3-3 3" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const NAV = [
  { icon: Home, label: "Overview" },
  { icon: Gauge, label: "Telemetry" },
  { icon: Chart, label: "Standings" },
  { icon: Cards, label: "Drivers" },
  { icon: Flag, label: "Strategy" },
  { icon: Calendar, label: "Calendar" },
  { icon: People, label: "Teams" },
];

export function Sidebar() {
  const [active, setActive] = useState(0);

  return (
    <aside className="sticky top-0 hidden h-screen w-[68px] flex-col items-center border-r border-line bg-[#0c0c12]/80 py-5 backdrop-blur-md md:flex">
      {/* logo */}
      <motion.div
        initial={{ scale: 0, rotate: -90 }}
        animate={{ scale: 1, rotate: 0 }}
        transition={{ type: "spring", stiffness: 220, damping: 16 }}
        className="mb-8 grid h-11 w-11 place-items-center rounded-2xl bg-gradient-to-br from-f1 to-[#9b0020] text-white shadow-glow"
      >
        <span className="text-[15px] font-extrabold tracking-tighter">F1</span>
      </motion.div>

      <nav className="flex flex-1 flex-col items-center gap-2">
        {NAV.map((item, i) => {
          const Ico = item.icon;
          const isActive = i === active;
          return (
            <button
              key={item.label}
              onClick={() => setActive(i)}
              title={item.label}
              className={`nav-icon ${isActive ? "nav-icon-active" : ""}`}
            >
              <Ico />
              {isActive && (
                <motion.span
                  layoutId="nav-dot"
                  className="absolute -left-[14px] h-5 w-[3px] rounded-full bg-f1"
                />
              )}
            </button>
          );
        })}
      </nav>

      <div className="mt-auto flex flex-col items-center gap-2">
        <button className="nav-icon" title="Settings">
          <Cog />
        </button>
        <button className="nav-icon" title="Log out">
          <Logout />
        </button>
        <div className="mt-2 h-9 w-9 overflow-hidden rounded-full bg-gradient-to-br from-[#3671C6] to-[#27F4D2] ring-2 ring-white/10" />
      </div>
    </aside>
  );
}

"use client";

import { motion } from "framer-motion";
import Image from "next/image";
import { NAV, type ViewId } from "@/lib/nav";

export function Sidebar({
  active,
  onChange,
}: {
  active: ViewId;
  onChange: (v: ViewId) => void;
}) {
  return (
    <aside className="sticky top-0 hidden h-screen w-[68px] flex-col items-center border-r border-line bg-base/80 py-5 backdrop-blur-md md:flex">
      {/* logo */}
      <motion.div
        initial={{ scale: 0, rotate: -90 }}
        animate={{ scale: 1, rotate: 0 }}
        transition={{ type: "spring", stiffness: 220, damping: 16 }}
        className="mb-8 grid h-11 w-11 place-items-center"
      >
        <Image
          src="/f1-logo.png"
          alt="F1"
          width={744}
          height={187}
          priority
          className="h-auto w-full"
        />
      </motion.div>

      <nav className="flex flex-1 flex-col items-center gap-2">
        {NAV.map((item) => {
          const Ico = item.icon;
          const isActive = item.id === active;
          return (
            <button
              key={item.id}
              onClick={() => onChange(item.id)}
              title={item.label}
              aria-label={item.label}
              aria-current={isActive ? "page" : undefined}
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
    </aside>
  );
}

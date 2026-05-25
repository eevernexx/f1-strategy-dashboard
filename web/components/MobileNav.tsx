"use client";

import { AnimatePresence, motion } from "framer-motion";
import Image from "next/image";
import { NAV, type ViewId } from "@/lib/nav";

export function MobileNav({
  open,
  active,
  onChange,
  onClose,
}: {
  open: boolean;
  active: ViewId;
  onChange: (v: ViewId) => void;
  onClose: () => void;
}) {
  return (
    <AnimatePresence>
      {open && (
        <div className="fixed inset-0 z-50 md:hidden">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
          />
          <motion.aside
            initial={{ x: -280 }}
            animate={{ x: 0 }}
            exit={{ x: -280 }}
            transition={{ type: "spring", stiffness: 320, damping: 32 }}
            className="absolute left-0 top-0 flex h-full w-[240px] flex-col border-r border-line bg-surface p-4"
          >
            <div className="mb-6 flex items-center gap-2.5">
              <span className="grid h-9 w-12 place-items-center">
                <Image src="/f1-logo.png" alt="F1" width={744} height={187} className="h-auto w-full" />
              </span>
              <span className="text-[14px] font-bold text-ink">F1 Dashboard Visualization</span>
            </div>

            <nav className="flex flex-col gap-1">
              {NAV.map((item) => {
                const Ico = item.icon;
                const isActive = item.id === active;
                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      onChange(item.id);
                      onClose();
                    }}
                    className={`flex items-center gap-3 rounded-xl px-3 py-2.5 text-[14px] font-medium transition-colors ${
                      isActive ? "bg-f1 text-white shadow-glow" : "text-ink-dim hover:bg-ov/[0.05] hover:text-ink"
                    }`}
                  >
                    <Ico />
                    {item.label}
                  </button>
                );
              })}
            </nav>
          </motion.aside>
        </div>
      )}
    </AnimatePresence>
  );
}

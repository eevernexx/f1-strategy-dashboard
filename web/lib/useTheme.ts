"use client";

import { useEffect, useState } from "react";

type Theme = "dark" | "light";

export function useTheme() {
  const [theme, setTheme] = useState<Theme>("dark");

  useEffect(() => {
    const saved = (typeof localStorage !== "undefined" &&
      localStorage.getItem("f1-theme")) as Theme | null;
    setTheme(saved === "light" ? "light" : "dark");
  }, []);

  const toggle = () => {
    setTheme((prev) => {
      const next: Theme = prev === "dark" ? "light" : "dark";
      document.documentElement.classList.toggle("light", next === "light");
      try {
        localStorage.setItem("f1-theme", next);
      } catch {}
      return next;
    });
  };

  return { theme, toggle };
}

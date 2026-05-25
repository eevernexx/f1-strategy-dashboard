import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Theme-driven surfaces (see CSS vars in globals.css)
        base: "var(--c-base)",
        surface: "var(--c-surface)",
        "surface-2": "var(--c-surface-2)",
        card: "var(--c-card)",
        line: "var(--c-line)",
        // neutral overlay (white in dark, near-black in light) with alpha support
        ov: "rgb(var(--c-ov) / <alpha-value>)",
        // F1 red accent (constant across themes)
        f1: {
          DEFAULT: "#E8002D",
          hover: "#FF1744",
          soft: "rgba(232,0,45,0.14)",
        },
        ink: {
          DEFAULT: "var(--c-ink)",
          dim: "var(--c-ink-dim)",
          faint: "var(--c-ink-faint)",
        },
        pos: "#22c55e",
        neg: "#ef4444",
      },
      fontFamily: {
        sans: ["var(--font-jakarta)", "system-ui", "sans-serif"],
      },
      borderRadius: {
        card: "20px",
        pill: "999px",
      },
      boxShadow: {
        card: "0 1px 0 rgba(255,255,255,0.04) inset, 0 12px 40px -12px rgba(0,0,0,0.6)",
        glow: "0 0 0 1px rgba(232,0,45,0.4), 0 8px 30px -6px rgba(232,0,45,0.45)",
      },
    },
  },
  plugins: [],
};

export default config;

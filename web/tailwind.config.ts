import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Finrise-style dark surfaces
        base: "#0a0a0f",
        surface: "#13131b",
        "surface-2": "#191922",
        card: "#15151e",
        line: "rgba(255,255,255,0.07)",
        // F1 red accent
        f1: {
          DEFAULT: "#E8002D",
          hover: "#FF1744",
          soft: "rgba(232,0,45,0.14)",
        },
        ink: {
          DEFAULT: "#f4f4f6",
          dim: "#a0a0b0",
          faint: "#6a6a78",
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

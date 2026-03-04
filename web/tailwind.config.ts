import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          primary: "#0B0F19",
          secondary: "#131825",
          card: "#1A1F2E",
        },
        accent: {
          gold: "#D4A017",
          "gold-light": "#E8C547",
          "gold-dark": "#B8860B",
        },
        signal: {
          buy: "#EF5350",
          sell: "#26A69A",
          hold: "#FFC107",
        },
        text: {
          primary: "#E2E4E9",
          secondary: "#8B90A0",
          muted: "#5A5F70",
        },
      },
      fontFamily: {
        sans: ["Noto Sans TC", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      backdropBlur: {
        xl: "20px",
      },
    },
  },
  plugins: [],
};

export default config;

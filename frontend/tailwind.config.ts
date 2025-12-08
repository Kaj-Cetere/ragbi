import type { Config } from "tailwindcss";

/**
 * Tailwind CSS v4 Configuration
 * Note: Most theme configuration is now done in globals.css using @theme directive.
 * This config file is minimal and only for content paths.
 */
const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  plugins: [],
};

export default config;
